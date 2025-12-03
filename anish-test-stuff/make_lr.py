from PIL import Image
import argparse

def make_lr():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the High-Res image')
    parser.add_argument('--scale', type=int, default=4, help='Downscale factor (matches your model training)')
    parser.add_argument('--output_path', type=str, default='test_imageLR.png', help='Output filename')
    args = parser.parse_args()

    # Open the High-Res Image
    img = Image.open(args.image_path).convert("RGB")
    
    # Calculate the new small size
    new_width = img.width // args.scale
    new_height = img.height // args.scale
    
    # Resize down using BICUBIC (This is the standard math used in SR research)
    lr_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    # Save it
    lr_img.save(args.output_path)
    
    print(f"Original Size: {img.width}x{img.height}")
    print(f"Low-Res Size:  {new_width}x{new_height}")
    print(f"Saved to:      {args.output_path}")

if __name__ == '__main__':
    make_lr()