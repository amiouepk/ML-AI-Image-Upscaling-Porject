import torch
from model import EDSR
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import os

def test():
    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the low-res image')
    parser.add_argument('--output_path', type=str, default='result.png', help='Where to save the result')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load the Model Architecture
    # Make sure scale_factor matches what you trained with (default was 4)
    model = EDSR(scale_factor=4).to(device)

    # 3. Load the Weights
    print(f"Loading weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval() # Set to evaluation mode (turns off dropout/batchnorm stuff if generic)

    # 4. Process the Image
    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found.")
        return

    # Open image
    img = Image.open(args.image_path).convert('RGB')
    
    # Convert to Tensor and send to GPU
    img_t = ToTensor()(img).unsqueeze(0).to(device) # unsqueeze adds Batch dimension (1, C, H, W)

    # 5. Run Inference
    with torch.no_grad():
        output_t = model(img_t)

    # 6. Save Result
    # Clamp ensures values don't go below 0 or above 1 (visual artifacts)
    output_t = output_t.squeeze(0).clamp(0, 1).cpu() 
    output_img = ToPILImage()(output_t)
    
    output_img.save(args.output_path)
    print(f"Super-Resolution image saved to: {args.output_path}")

if __name__ == "__main__":
    test()