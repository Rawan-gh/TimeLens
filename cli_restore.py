
import os
import argparse
import urllib.request
import numpy as np
from PIL import Image
from gfpgan import GFPGANer

GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MODELS_DIR = "models"

def ensure_weights():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "GFPGANv1.3.pth")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(GFPGAN_MODEL_URL, model_path)
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Restore faces in an image using GFPGAN.")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    args = parser.parse_args()

    model_path = ensure_weights()
    restorer = GFPGANer(model_path=model_path, upscale=2, arch="clean", channel_multiplier=2, bg_upsampler=None)

    img = Image.open(args.input).convert("RGB")
    bgr = np.array(img)[:, :, ::-1]
    _, _, restored_img = restorer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)

    if restored_img is None:
        img.save(args.output)
        return

    rgb = restored_img[:, :, ::-1].astype("uint8")
    Image.fromarray(rgb).save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
