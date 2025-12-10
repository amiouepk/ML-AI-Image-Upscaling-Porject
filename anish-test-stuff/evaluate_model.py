import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

def calculate_psnr(img1, img2):
    """Calculates PSNR (Peak Signal-to-Noise Ratio)."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def get_y_channel(img):
    """Converts BGR image to YCbCr and extracts the Y channel."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return img

def crop_to_match(img1, img2):
    """Crops two images to the minimum common dimensions."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]

def evaluate(hr_path, bicubic_path, upscaled_path):
    print(f"\nLOADING IMAGES:")
    print(f"1. HR (Ground Truth): {hr_path}")
    print(f"2. Bicubic (Baseline): {bicubic_path}")
    print(f"3. EDSR (Your Model):  {upscaled_path}")

    # Load images
    img_hr = cv2.imread(hr_path)
    img_bic = cv2.imread(bicubic_path)
    img_sr = cv2.imread(upscaled_path)

    # Check if images loaded successfully
    if img_hr is None: sys.exit(f"Error: Could not open HR image: {hr_path}")
    if img_bic is None: sys.exit(f"Error: Could not open Bicubic image: {bicubic_path}")
    if img_sr is None: sys.exit(f"Error: Could not open Upscaled image: {upscaled_path}")

    # --- 1. Compare HR vs Bicubic ---
    # align shapes
    crop_hr, crop_bic = crop_to_match(img_hr, img_bic)
    # convert to Y channel
    y_hr = get_y_channel(crop_hr)
    y_bic = get_y_channel(crop_bic)
    
    psnr_bic = calculate_psnr(y_hr, y_bic)
    ssim_bic = ssim(y_hr, y_bic, data_range=255)

    # --- 2. Compare HR vs EDSR ---
    # align shapes
    crop_hr_2, crop_sr = crop_to_match(img_hr, img_sr)
    # convert to Y channel
    y_hr_2 = get_y_channel(crop_hr_2)
    y_sr = get_y_channel(crop_sr)

    psnr_edsr = calculate_psnr(y_hr_2, y_sr)
    ssim_edsr = ssim(y_hr_2, y_sr, data_range=255)

    # --- 3. Output Results ---
    print("-" * 60)
    print(f"{'METRIC':<10} | {'BICUBIC':<15} | {'EDSR (YOURS)':<15} | {'IMPROVEMENT'}")
    print("-" * 60)
    
    # PSNR Row
    diff_psnr = psnr_edsr - psnr_bic
    print(f"{'PSNR':<10} | {psnr_bic:<15.4f} | {psnr_edsr:<15.4f} | {diff_psnr:+.4f} dB")
    
    # SSIM Row
    diff_ssim = ssim_edsr - ssim_bic
    print(f"{'SSIM':<10} | {ssim_bic:<15.4f} | {ssim_edsr:<15.4f} | {diff_ssim:+.4f}")
    print("-" * 60)

    if diff_psnr > 0:
        print("SUCCESS: Your model outperforms Bicubic interpolation!")
    else:
        print("RESULT: Your model is currently underperforming Bicubic.")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR and SSIM for SR models.")
    parser.add_argument("hr", help="Path to Original High-Res image")
    parser.add_argument("bicubic", help="Path to Bicubic image")
    parser.add_argument("upscaled", help="Path to Model Upscaled image")
    
    args = parser.parse_args()
    
    evaluate(args.hr, args.bicubic, args.upscaled)