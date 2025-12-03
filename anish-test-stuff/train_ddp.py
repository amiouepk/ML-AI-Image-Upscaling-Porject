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
    def __init__(self, root_dir, patch_size=96, scale_factor=4, repeat=40):
        super(UniversalDataset, self).__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.repeat = repeat
        
        # Recursively find all images in the directory
        self.image_files = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Walk through folder and subfolders
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_files.append(os.path.join(root, file))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.hr_transform = Compose([
            RandomCrop(patch_size),
            ToTensor()
        ])
        
        # Normalize (Mean subtraction for EDSR)
        self.normalize = Normalize(mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0])

    def __getitem__(self, index):
        # Modulo index to allow "repeating" the dataset
        actual_index = index % len(self.image_files)
        
        try:
            hr_path = self.image_files[actual_index]
            hr_image = Image.open(hr_path).convert("RGB")
            
            # Create HR Tensor (Cropped)
            hr_tensor = self.hr_transform(hr_image)
            
            # Create LR Image (Downscaled)
            lr_image = F.resize(ToPILImage()(hr_tensor), 
                              hr_tensor.shape[1] // self.scale_factor, 
                              interpolation=Image.BICUBIC)
            lr_tensor = ToTensor()(lr_image)
            
            # Apply Normalization
            lr_tensor = self.normalize(lr_tensor)
            hr_tensor = self.normalize(hr_tensor)
            
            return lr_tensor, hr_tensor
        except Exception as e:
            # Skip bad images instead of crashing
            print(f"Warning: Error loading {hr_path}: {e}")
            # Recursively try the next one
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.image_files) * self.repeat

# --- 2. Kaggle Downloader ---
def download_kaggle_dataset(dataset_name, save_dir):
    """
    Downloads and unzips a Kaggle dataset.
    Example dataset_name: 'joe1995/div2k-dataset'
    """
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Data seems to exist in {save_dir}. Skipping download.")
        return

    print(f"Downloading {dataset_name} from Kaggle to {save_dir}...")
    
    # Ensure folder exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Use Kaggle CLI API
    # We use subprocess because installing the python lib on compute nodes can be finicky
    try:
        cmd = f"kaggle datasets download -d {dataset_name} -p {save_dir} --unzip"
        subprocess.check_call(cmd, shell=True)
        print("Download complete.")
    except subprocess.CalledProcessError:
        print("Error: Kaggle download failed.")
        print("1. Make sure you have ~/.kaggle/kaggle.json")
        print("2. Make sure you have internet access (try running on login node first)")
        raise

# --- 3. Distributed Setup ---
def setup_distributed():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Resolve Master Addr for Perlmutter
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
        # Fallback for local testing
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
    # Example Kaggle Dataset: 'joe1995/div2k-dataset' or 'soumikrakshit/flickr8k'
    kaggle_dataset_name = 'joe1995/div2k-dataset' 
    
    # Where to save data on Scratch
    data_root = '/pscratch/sd/a/am3138/datasets/DIV2K' 
    
    batch_size = 16
    num_epochs = 20 # Lower epochs because of 'repeat' logic
    learning_rate = 1e-4

    # --- DOWNLOAD DATA (Rank 0 only) ---
    if rank == 0:
        try:
            download_kaggle_dataset(kaggle_dataset_name, data_root)
        except Exception as e:
            print(f"Download failed: {e}")
            # We don't exit, we hope data is there. If not, dataset will crash later.
    
    # Wait for Rank 0 to finish downloading before other GPUs start
    if dist.is_initialized():
        dist.barrier()

    # --- DATA LOADING ---
    # UniversalDataset will find images recursively in data_root
    train_dataset = UniversalDataset(root_dir=data_root, repeat=40)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    # --- MODEL SETUP ---
    model = EDSR().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    if rank == 0:
        print(f"Training on {len(train_dataset)} patches per epoch (Repeat=40)")

    # --- LOOP ---
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
            optimizer.step()

            running_loss += loss.item()
            
            # Optional: Print progress every 100 batches
            if rank == 0 and i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_dataloader)

        if rank == 0:
            print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
            # Save checkpoint
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, f'edsr_epoch_{epoch+1}.pth')

    cleanup()

if __name__ == '__main__':
    main()