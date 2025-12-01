import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os
import math
from model import EDSR
import zipfile

# Assuming you have a custom dataset class 'SRDataset' that loads HR images
# and creates LR images on the fly (e.g., by Bicubic downsampling)

# --- Placeholder for a simple Dataset ---
class Urban100Dataset(Dataset):
    def __init__(self, data_root="data/urban100", scale_factor=4):
        self.scale = scale_factor
        self.data_root = data_root
        self.hr_dir = os.path.join(data_root, "Urban100")

        # --- Auto-download if dataset not present ---
        if not os.path.exists(self.hr_dir):
            print("Urban100 dataset not found. Downloading from Kaggle...")
            self._download_from_kaggle()

        # --- Load file list ---
        self.hr_files = sorted([
            os.path.join(self.hr_dir, f)
            for f in os.listdir(self.hr_dir)
            if f.lower().endswith((".png", ".jpg"))
        ])

    def _download_from_kaggle(self):
        os.makedirs(self.data_root, exist_ok=True)

        # Kaggle command → requires Kaggle API token present in ~/.kaggle/kaggle.json
        kaggle.api.dataset_download_files(
            "harshraone/urban100",
            path=self.data_root,
            unzip=True
        )

        print("Download complete!")

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        w, h = hr.size

        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        hr = hr.crop((0, 0, (w // self.scale) * self.scale,
                             (h // self.scale) * self.scale))

        return TF.to_tensor(lr), TF.to_tensor(hr)

    def __len__(self):
        return len(self.hr_files)
    
# --- Main Training Script ---

def train_edsr(scale_factor=4, n_resblocks=16, n_feats=64, epochs=100, batch_size=16, lr=1e-4):
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Model Initialization (Import EDSR from your model.py)
    # Assuming EDSR class is available (e.g., from model.py)
    model = EDSR(scale_factor=scale_factor, n_resblocks=n_resblocks, n_feats=n_feats).to(device)

    # 3. Multi-GPU Parallelization (If available)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model) # Simple and effective for training

    # 4. Data Loading (Replace 'path/to/hr/images' with your actual directory)
    train_dataset = SimpleSRDataset(scale_factor=scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 5. Loss Function and Optimizer
    # L1 Loss is standard and better than MSE for visually sharp results in SR
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for lr_images, hr_images in train_loader:
            # Move data to GPU
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            sr_images = model(lr_images)

            # Loss calculation
            loss = criterion(sr_images, hr_images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
        #validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for lr_images, hr_images in train_loader:   # you can create a separate val loader later
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
    
                sr_images = model(lr_images)
                val_loss += criterion(sr_images, hr_images).item()
    
        print(f"Validation Loss: {val_loss/len(train_loader):.4f}")
        
        # Save a checkpoint occasionally
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), f'edsr_x{scale_factor}_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_edsr()
