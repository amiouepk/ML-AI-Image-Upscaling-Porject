import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.transforms import ToTensor, Compose, RandomCrop, ToPILImage
import torchvision.transforms.functional as F
from PIL import Image
import os
from glob import glob
import argparse
import subprocess

# Import your model (assuming model.py is in the same folder)
from model import EDSR

# --- Custom Dataset ---
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale_factor=4):
        super(DIV2KDataset, self).__init__()
        self.image_files = sorted(glob(os.path.join(hr_dir, '*.png')))
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.hr_transform = Compose([
            RandomCrop(patch_size),
            ToTensor()
        ])

    def __getitem__(self, index):
        hr_path = self.image_files[index]
        hr_image = Image.open(hr_path).convert("RGB")
        hr_tensor = self.hr_transform(hr_image)
        # Create LR image
        lr_image = F.resize(ToPILImage()(hr_tensor), hr_tensor.shape[1] // self.scale_factor, interpolation=Image.BICUBIC)
        lr_tensor = ToTensor()(lr_image)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.image_files)

def setup_distributed():
    """
    Setup distributed training for NERSC Perlmutter (SLURM).
    """
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        # --- ADD THIS BLOCK ---
        import subprocess
        try:
            cmd = "scontrol show hostnames $SLURM_JOB_NODELIST"
            stdout = subprocess.check_output(cmd, shell=True)
            hostnames = stdout.decode().splitlines()
            os.environ['MASTER_ADDR'] = hostnames[0]
            os.environ['MASTER_PORT'] = "29500"
        except Exception as e:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = "29500"
        # ----------------------

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, world_size, local_rank
    else:
        print("Not using SLURM, falling back to CPU or single GPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, 0

def cleanup():
    dist.destroy_process_group()

def main():
    # --- 1. Setup Distributed Environment ---
    device, rank, world_size, local_rank = setup_distributed()
    
    # Only print on the master process (Rank 0) to avoid messy logs
    if rank == 0:
        print(f"Starting training on {world_size} GPUs.")

    # --- 2. Configuration ---
    num_epochs = 100
    # Batch size per GPU. If set to 16 and you have 4 GPUs, effective batch size is 64.
    batch_size = 16 
    learning_rate = 1e-4

    # UPDATE THESE PATHS TO YOUR SCRATCH DIRECTORY
    # Example: /pscratch/sd/u/username/DIV2K_train_HR
    train_hr_dir = '/pscratch/sd/a/am3138/DIV2K_train_HR' 
    val_hr_dir = '/pscratch/sd/a/am3138/DIV2K_valid_HR'

    if rank == 0:
        if not os.path.exists(train_hr_dir):
            print(f"Error: Path not found {train_hr_dir}")
            return

    # --- 3. Data Loading ---
    train_dataset = DIV2KDataset(hr_dir=train_hr_dir)
    
    # DistributedSampler is CRITICAL. It ensures each GPU gets different data.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # num_workers should be set (4-8 is usually good on Perlmutter)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    # Validation (Optional: usually just run val on rank 0 to save time, or use DistributedSampler without shuffle)
    val_dataset = DIV2KDataset(hr_dir=val_hr_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # --- 4. Model Setup ---
    model = EDSR().to(device)
    
    # Wrap model in DDP
    # device_ids tells DDP which GPU this process uses
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # --- 5. Training Loop ---
    for epoch in range(num_epochs):
        # IMPORTANT: Set epoch for sampler to ensure shuffling works correctly across epochs
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        for lr_imgs, hr_imgs in train_dataloader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss across the epoch
        avg_loss = running_loss / len(train_dataloader)

        # Validation & Saving (Only on Rank 0)
        if rank == 0:
            # Simple validation logic
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for lr_v, hr_v in val_dataloader:
                    lr_v = lr_v.to(device)
                    hr_v = hr_v.to(device)
                    sr_v = model(lr_v)
                    val_loss += criterion(sr_v, hr_v).item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save checkpoint
            torch.save(model.module.state_dict(), f'edsr_epoch_{epoch+1}.pth') 
            # Note: use model.module.state_dict() to remove the "module." prefix added by DDP

    cleanup()

if __name__ == '__main__':
    main()
