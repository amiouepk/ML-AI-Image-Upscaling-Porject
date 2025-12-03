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
    def __init__(self, root_dir, patch_size=144, scale_factor=4, repeat=1):
        super(UniversalDataset, self).__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.repeat = repeat
        
        self.image_files = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Walk through folder
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_files.append(os.path.join(root, file))

        if len(self.image_files) == 0:
            # Fallback for validation: if valid folder is empty/missing, don't crash immediately
            print(f"Warning: No images found in {root_dir}")

        self.hr_transform = Compose([
            RandomCrop(patch_size),
            ToTensor()
        ])
        
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

# --- 2. Kaggle Downloader ---
def download_kaggle_dataset(dataset_name, save_dir):
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Data seems to exist in {save_dir}. Skipping download.")
        return
    
    print(f"Downloading {dataset_name} from Kaggle to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    try:
        cmd = f"kaggle datasets download -d {dataset_name} -p {save_dir} --unzip"
        subprocess.check_call(cmd, shell=True)
        print("Download complete.")
    except subprocess.CalledProcessError:
        print("Error: Kaggle download failed.")
        raise

# --- 3. Distributed Setup ---
def setup_distributed():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        if rank == 0:
            print(f"SLURM_NODELIST: {os.environ['SLURM_NODELIST']}")
        
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
        print("Not using SLURM. Running in single process mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 4. Main Training Loop ---
def main():
    device, rank, world_size, local_rank = setup_distributed()

    # --- CONFIGURATION ---
    kaggle_dataset_name = 'joe1995/div2k-dataset' 
    data_root = '/pscratch/sd/a/am3138/datasets/DIV2K' 
    
    # Specific paths for Train and Validation
    train_dir = os.path.join(data_root, 'DIV2K_train_HR')
    val_dir = os.path.join(data_root, 'DIV2K_valid_HR')

    # Optimization Config
    batch_size = 32 
    patch_size = 144
    num_epochs = 20 
    learning_rate = 1e-4

    # --- DOWNLOAD (Rank 0 only) ---
    if rank == 0:
        try:
            download_kaggle_dataset(kaggle_dataset_name, data_root)
        except Exception as e:
            print(f"Download failed: {e}")
    
    if dist.is_initialized():
        dist.barrier()

    # --- DATA LOADING ---
    # 1. Training Set (High Repeat for learning)
    if not os.path.exists(train_dir): train_dir = data_root # Fallback
    train_dataset = UniversalDataset(root_dir=train_dir, patch_size=patch_size, repeat=40)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                  num_workers=16, pin_memory=True, prefetch_factor=4)

    # 2. Validation Set (No Repeat, Shuffle=False)
    # If val dir doesn't exist, use train dir but repeat=1 just to check code doesn't crash
    if not os.path.exists(val_dir): val_dir = train_dir 
    
    val_dataset = UniversalDataset(root_dir=val_dir, patch_size=patch_size, repeat=1)
    # We use DistributedSampler for Validation too so all 4 GPUs split the work
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8)

    # --- MODEL ---
    model = EDSR(scale_factor=4, n_resblocks=32, n_feats=96).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)
    criterion = nn.L1Loss()

    if rank == 0:
        print(f"Training on {len(train_dataset)} patches. Validating on {len(val_dataset)} patches.")

    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        # Train
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
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_dataloader)}] Train Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_dataloader)

        # --- VALIDATION LOOP ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_dataloader:
                lr_v = lr_v.to(device, non_blocking=True)
                hr_v = hr_v.to(device, non_blocking=True)
                
                sr_v = model(lr_v)
                loss_v = criterion(sr_v, hr_v)
                running_val_loss += loss_v.item()

        # Calculate average validation loss across this GPU's slice
        local_val_avg = running_val_loss / len(val_dataloader)
        
        # Convert to tensor to sync across GPUs
        val_loss_tensor = torch.tensor(local_val_avg).to(device)
        
        # Sum up the averages from all GPUs and divide by world_size
        if dist.is_initialized():
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            global_val_loss = val_loss_tensor.item() / world_size
        else:
            global_val_loss = local_val_avg

        # --- PRINT & SAVE ---
        if rank == 0:
            print(f"="*50)
            print(f"Epoch {epoch+1} Completed.")
            print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {global_val_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"="*50)
            
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, f'edsr_epoch_{epoch+1}.pth')

    cleanup()

if __name__ == '__main__':
    main()