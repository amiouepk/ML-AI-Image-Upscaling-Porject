import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os
import math

# Assuming you have a custom dataset class 'SRDataset' that loads HR images
# and creates LR images on the fly (e.g., by Bicubic downsampling)

# --- Placeholder for a simple Dataset ---
class SimpleSRDataset(Dataset):
    def __init__(self, hr_dir, scale_factor):
        # In a real scenario, this loads a list of HR image paths
        self.hr_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]
        self.scale = scale_factor
        self.to_tensor = TF.to_tensor

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # 1. Load HR image
        hr_img = Image.open(self.hr_files[idx]).convert('RGB')
        
        # 2. Downsample to create LR image (using Bicubic is standard)
        w, h = hr_img.size
        lr_w, lr_h = w // self.scale, h // self.scale
        lr_img = hr_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
        
        # 3. Crop HR to ensure its size is a multiple of scale_factor
        hr_img = hr_img.crop((0, 0, lr_w * self.scale, lr_h * self.scale))

        # 4. Convert to PyTorch Tensor
        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        return lr_tensor, hr_tensor
    
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
    train_dataset = SimpleSRDataset(hr_dir='path/to/hr/images', scale_factor=scale_factor)
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
        
        # Save a checkpoint occasionally
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), f'edsr_x{scale_factor}_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_edsr()