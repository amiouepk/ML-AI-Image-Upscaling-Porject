import torch
from model import EDSR
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import os

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the low-res image')
    parser.add_argument('--output_path', type=str, default='test_output.png', help='Where to save the result')
    # Add scale argument in case you want to test 2x or 4x
    parser.add_argument('--scale', type=int, default=4) 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model (Make sure parameters match training!)
    # We used n_resblocks=32 and n_feats=96 in the optimized run
    model = EDSR(scale_factor=args.scale, n_resblocks=32, n_feats=96).to(device)

    # 2. Load Weights
    print(f"Loading weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Prepare Image
    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found.")
        return

    img = Image.open(args.image_path).convert('RGB')
    img_t = ToTensor()(img).unsqueeze(0).to(device)

    # --- NORMALIZATION FIX ---
    # We must normalize the input just like we did in training
    mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1).to(device)
    img_t = img_t - mean
    # -------------------------

    # 4. Inference
    with torch.no_grad():
        output_t = model(img_t)

    # --- DENORMALIZATION FIX ---
    # Add the mean back to get correct colors
    output_t = output_t + mean
    # ---------------------------

    # 5. Save
    output_t = output_t.squeeze(0).clamp(0, 1).cpu() 
    output_img = ToPILImage()(output_t)
    output_img.save(args.output_path)
    print(f"Saved SR image to: {args.output_path}")

if __name__ == "__main__":
    test()