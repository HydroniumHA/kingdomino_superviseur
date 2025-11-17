import os
import cv2
import numpy as np
import json
import argparse

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


def mean_hsv_of_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque : ignorer pixels très sombres (fond noir) et pixels transparents si présents
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = (v > 10) & (s > 10)

    if not np.any(mask):
        # fallback: use all pixels
        mask = np.ones_like(v, dtype=bool)

    # cv2.mean supports mask but expects single-channel mask of type uint8
    mask_u8 = (mask.astype(np.uint8) * 255)
    mean = cv2.mean(hsv, mask=mask_u8)[:3]

    # also return mean BGR for convenience
    mean_bgr = cv2.mean(img, mask=mask_u8)[:3]

    # return integer lists
    return {
        'mean_hsv': [int(round(x)) for x in mean],
        'mean_bgr': [int(round(x)) for x in mean_bgr]
    }


def build_palette_from_dir(directory):
    out = {}
    for fname in sorted(os.listdir(directory)):
        _, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        path = os.path.join(directory, fname)
        info = mean_hsv_of_image(path)
        if info is None:
            print(f"Warning: failed to load {fname}")
            continue
        out[fname] = info
        print(f"Processed {fname}: HSV={info['mean_hsv']}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Compute mean HSV for sample tile images in a folder')
    parser.add_argument('--dir', '-d', default='data/clear_images', help='Directory containing tile sample images')
    parser.add_argument('--out', '-o', default='biome_palette_samples.json', help='Output JSON file')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: directory not found: {args.dir}")
        return

    palette = build_palette_from_dir(args.dir)
    if not palette:
        print('No images processed. Check files in the directory.')
        return

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(palette, f, indent=2, ensure_ascii=False)

    print(f"Saved palette to {args.out} ({len(palette)} entries)")


if __name__ == '__main__':
    main()
