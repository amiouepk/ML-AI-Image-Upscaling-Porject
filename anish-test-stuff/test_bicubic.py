from PIL import Image
import argparse
import os

def test_bicubic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the Low-Res input image')
    parser.add_argument('--output_path', type=str, default='result_bicubic.png', help='Path to save the result')
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor (must match your model, usually 4)')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Could not find image {args.image_path}")
        return

    # 1. Load the Low-Res Image
    lr_img = Image.open(args.image_path).convert("RGB")
    print(f"Low-Res Input Size: {lr_img.width}x{lr_img.height}")

    # 2. Calculate Target Size
    target_width = lr_img.width * args.scale
    target_height = lr_img.height * args.scale

    # 3. Perform Bicubic Interpolation
    # This is the standard mathematical method for resizing images
    bicubic_img = lr_img.resize((target_width, target_height), Image.BICUBIC)

    # 4. Save
    bicubic_img.save(args.output_path)
    print(f"Bicubic Output Size: {bicubic_img.width}x{bicubic_img.height}")
    print(f"Saved to: {args.output_path}")

if __name__ == '__main__':
    test_bicubic()