import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.transforms import ToTensor, Compose, RandomCrop, ToPILImage, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import os
import argparse
import subprocess

# Import your model
from model import EDSR

# --- 1. Universal Dataset Class ---
class UniversalDataset(Dataset):
    def __init__(self, root_dir, patch_size=192, scale_factor=4, repeat=10):
        super(UniversalDataset, self).__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.repeat = repeat
        
        self.image_files = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        # Recursively walk through ALL subfolders
        print(f"Scanning {root_dir} for images...")
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_extensions:
                        self.image_files.append(os.path.join(root, file))

        if len(self.image_files) == 0:
            print(f"Warning: No images found in {root_dir}")

        self.hr_transform = Compose([
            RandomCrop(patch_size),
            ToTensor()
        ])
        
        # EDSR Mean Normalization
        self.normalize = Normalize(mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0])

    def __getitem__(self, index):
        if len(self.image_files) == 0: return torch.zeros(1), torch.zeros(1)

        actual_index = index % len(self.image_files)
        
        try:
            hr_path = self.image_files[actual_index]
            hr_image = Image.open(hr_path).convert("RGB")
            
            hr_tensor = self.hr_transform(hr_image)
            
            lr_image = F.resize(ToPILImage()(hr_tensor), 
                              hr_tensor.shape[1] // self.scale_factor, 
                              interpolation=Image.BICUBIC)
            lr_tensor = ToTensor()(lr_image)
            
            lr_tensor = self.normalize(lr_tensor)
            hr_tensor = self.normalize(hr_tensor)
            
            return lr_tensor, hr_tensor
        except Exception as e:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.image_files) * self.repeat

# --- 2. Distributed Setup ---
def setup_distributed():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        try:
            cmd = "scontrol show hostnames $SLURM_JOB_NODELIST"
            stdout = subprocess.check_output(cmd, shell=True)
            hostnames = stdout.decode().splitlines()
            os.environ['MASTER_ADDR'] = hostnames[0]
            os.environ['MASTER_PORT'] = "29500"
        except:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = "29500"

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, world_size, local_rank
    else:
        print("Single GPU Mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 3. Main ---
def main():
    device, rank, world_size, local_rank = setup_distributed()

    # --- PATH CONFIGURATION ---
    data_root = '/pscratch/sd/a/am3138/datasets_combined/DF2K'
    
    # Path logic checks
    possible_val_paths = [
        os.path.join(data_root, 'DIV2K_valid_HR'),
        os.path.join(data_root, 'valid'),
        os.path.join(data_root, 'DF2K_valid_HR')
    ]
    
    val_dir = data_root 
    for p in possible_val_paths:
        if os.path.exists(p):
            val_dir = p
            break
            
    if rank == 0:
        print(f"Training Data Root: {data_root}")
        print(f"Validation Data:    {val_dir}")

    # --- HYPERPARAMETERS ---
    batch_size = 64     
    patch_size = 192    
    # CHANGE 1: Increased Epochs to 40
    num_epochs = 40
    learning_rate = 1e-4

    # --- DATA LOADING ---
    train_dataset = UniversalDataset(root_dir=data_root, patch_size=patch_size, repeat=10)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                  num_workers=30, pin_memory=True, prefetch_factor=2)

    # Validation
    if not os.path.exists(val_dir): val_dir = train_dir 
    val_dataset = UniversalDataset(root_dir=val_dir, patch_size=patch_size, repeat=1)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8)

    # --- MODEL ---
    model = EDSR(scale_factor=4, n_resblocks=32, n_feats=96).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # CHANGE 2: Adjusted Milestones to [20, 30] for longer training
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5)
    
    criterion = nn.L1Loss()

    if rank == 0:
        print(f"Total Training Samples per Epoch: {len(train_dataset)}")
        print(f"Total Epochs: {num_epochs} (LR decay at 20 and 30)")

    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, (lr_imgs, hr_imgs) in enumerate(train_dataloader):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            
            if rank == 0 and i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_dataloader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        
        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_dataloader:
                lr_v = lr_v.to(device, non_blocking=True)
                hr_v = hr_v.to(device, non_blocking=True)
                sr_v = model(lr_v)
                running_val_loss += criterion(sr_v, hr_v).item()

        local_val_avg = running_val_loss / len(val_dataloader)
        val_tensor = torch.tensor(local_val_avg).to(device)
        if dist.is_initialized():
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            global_val_loss = val_tensor.item() / world_size
        else:
            global_val_loss = local_val_avg

        if rank == 0:
            avg_train_loss = running_loss / len(train_dataloader)
            print(f"="*50)
            print(f"Epoch {epoch+1} Done. Train Loss: {avg_train_loss:.4f} | Val Loss: {global_val_loss:.4f}")
            print(f"="*50)
            
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, f'edsr_df2k_epoch_{epoch+1}.pth')

    cleanup()

if __name__ == '__main__':
    main()