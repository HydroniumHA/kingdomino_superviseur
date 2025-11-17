import os
import json
import cv2
import numpy as np
import argparse

DEFAULT_PALETTE = 'biome_palette_samples.json'


def mean_hsv_of_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = (v > 10) & (s > 10)
    if not np.any(mask):
        mask = np.ones_like(v, dtype=bool)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mean = cv2.mean(hsv, mask=mask_u8)[:3]
    mean = [float(x) for x in mean]
    return mean


def load_samples(path):
    """Load a samples palette file: filename -> {mean_hsv, mean_bgr}.
    This function only accepts the samples format (produced by `palette_from_samples.py`).
    If the provided file looks like centroids (hsv_mean keys), raises an error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Palette file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        return {}

    # quick check to ensure samples format
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict) and 'mean_hsv' in first_val:
        return data

    # If file appears to be centroids (hsv_mean), refuse — user asked to ignore centroids
    if isinstance(first_val, dict) and 'hsv_mean' in first_val:
        raise ValueError(f"Provided file {path} looks like centroids; expected samples format (mean_hsv).")

    raise ValueError(f"Unrecognized palette file format for {path}; expected samples mapping filename->{{mean_hsv,mean_bgr}}.")


def load_palette_or_centroids(path):
    """Load file which can be either:
    - a centroids file (biome -> {hsv_mean, bgr_mean})
    - a samples palette (filename -> {mean_hsv, mean_bgr})
    Returns tuple (mode, data) where mode is 'centroids' or 'samples'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Palette/centroids file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        return 'samples', {}

    first_val = next(iter(data.values()))
    # If this looks like centroids file (has 'hsv_mean'), convert to sample-like mapping
    if isinstance(first_val, dict) and 'hsv_mean' in first_val:
        samples = {}
        for biome, info in data.items():
            fname = f"{biome}_centroid"
            samples[fname] = {'mean_hsv': info['hsv_mean'], 'mean_bgr': info['bgr_mean']}
        return 'samples', samples

    # Otherwise assume samples mapping already (filename -> mean_hsv/mean_bgr)
    return 'samples', data


def infer_biome_name_from_filename(filename):
    base = os.path.splitext(filename)[0]
    biome = ''.join([c for c in base if not c.isdigit()]).rstrip('_').rstrip('-').strip()
    return biome or base


def classify_against_samples(mean_hsv, samples, mode='hsv'):
    """Compare mean_hsv against each sample in samples dict (filename -> {mean_hsv, mean_bgr}).
    Returns (best_filename, best_biome, distance)
    """
    mean = np.array(mean_hsv, dtype=float)
    best = None
    best_d = float('inf')
    best_file = None

    for fname, info in samples.items():
        sample_h = np.array(info.get('mean_hsv', [0,0,0]), dtype=float)
        sample_b = np.array(info.get('mean_bgr', [0,0,0]), dtype=float)

        if mode == 'hsv':
            d = np.linalg.norm(mean - sample_h)
        elif mode == 'bgr':
            # convert mean_hsv to bgr
            hsv_px = np.uint8([[[int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))]]])
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0,0].astype(float)
            d = np.linalg.norm(bgr_px - sample_b)
        elif mode == 'both':
            d_hsv = np.linalg.norm(mean - sample_h)
            hsv_px = np.uint8([[[int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))]]])
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0,0].astype(float)
            d_bgr = np.linalg.norm(bgr_px - sample_b)
            d = d_hsv + (d_bgr * 0.2)
        else:
            raise ValueError('mode must be hsv, bgr, or both')

        if d < best_d:
            best_d = d
            best = infer_biome_name_from_filename(fname)
            best_file = fname

    return best_file, best, best_d


def classify(mean_hsv, centroids, mode='hsv'):
    best = None
    best_d = float('inf')
    mean = np.array(mean_hsv, dtype=float)
    for biome, c in centroids.items():
        if mode == 'hsv':
            ref = np.array(c['hsv'], dtype=float)
        elif mode == 'bgr':
            ref = np.array(c['bgr'], dtype=float)
        elif mode == 'both':
            # combine HSV and BGR distances (scale BGR to similar range)
            hsv_ref = np.array(c['hsv'], dtype=float)
            bgr_ref = np.array(c['bgr'], dtype=float)
            d_hsv = np.linalg.norm(mean - hsv_ref)
            # compute mean_bgr of image by converting mean_hsv back to BGR roughly
            # we'll compute image mean BGR by converting a 1x1 HSV to BGR
            hsv_px = np.uint8([[[int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))]]])
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0,0].astype(float)
            d_bgr = np.linalg.norm(bgr_px - bgr_ref)
            d = d_hsv + (d_bgr * 0.2)
            if d < best_d:
                best_d = d
                best = biome
            continue
        else:
            raise ValueError('mode must be hsv, bgr, or both')

        d = np.linalg.norm(mean - ref)
        if d < best_d:
            best_d = d
            best = biome
    return best, best_d


def main():
    parser = argparse.ArgumentParser(description='Classify a tile image into a biome using sample palette')
    parser.add_argument('image', help='Path to input image to classify')
    parser.add_argument('--palette', '-p', default=DEFAULT_PALETTE, help='Path to palette JSON (from samples)')
    parser.add_argument('--mode', choices=['hsv','bgr','both'], default='hsv', help='Distance space to use')
    args = parser.parse_args()

    try:
        file_mode, data = load_palette_or_centroids(args.palette)
    except FileNotFoundError as e:
        print(e)
        return

    mean_hsv = mean_hsv_of_image(args.image)
    if mean_hsv is None:
        print(f"Error: failed to load image: {args.image}")
        return

    print(f"Image: {args.image}")
    print(f"Mean HSV: {[int(round(x)) for x in mean_hsv]}")

    if file_mode == 'centroids':
        biome, dist = classify(mean_hsv, data, mode=args.mode)
        print(f"Predicted biome: {biome} (distance={dist:.2f}, mode={args.mode})")
    else:
        best_file, biome, dist = classify_against_samples(mean_hsv, data, mode=args.mode)
        print(f"Best matching sample: {best_file}")
        print(f"Predicted biome: {biome} (distance={dist:.2f}, mode={args.mode})")


if __name__ == '__main__':
    main()
