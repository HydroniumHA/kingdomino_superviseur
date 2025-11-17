import json
import os
import numpy as np

INPUT = 'biome_palette_samples.json'
OUTPUT = 'biome_centroids.json'


def load_palette(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def infer_biome_name(filename):
    base = os.path.splitext(filename)[0]
    # remove trailing digits and separators
    biome = ''.join([c for c in base if not c.isdigit()])
    biome = biome.rstrip('_').rstrip('-').strip()
    return biome or base


def compute_centroids(palette):
    groups = {}
    for fname, info in palette.items():
        biome = infer_biome_name(fname)
        groups.setdefault(biome, {'hsv': [], 'bgr': []})
        groups[biome]['hsv'].append(info['mean_hsv'])
        groups[biome]['bgr'].append(info['mean_bgr'])

    centroids = {}
    for biome, lists in groups.items():
        hsv = np.array(lists['hsv'], dtype=float)
        bgr = np.array(lists['bgr'], dtype=float)
        centroids[biome] = {
            'count': int(hsv.shape[0]),
            'hsv_mean': hsv.mean(axis=0).round().astype(int).tolist(),
            'bgr_mean': bgr.mean(axis=0).round().astype(int).tolist()
        }
    return centroids


def main():
    try:
        palette = load_palette(INPUT)
    except FileNotFoundError:
        print(f"Palette file not found: {INPUT}")
        return

    centroids = compute_centroids(palette)

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(centroids, f, indent=2, ensure_ascii=False)

    print(f"Saved centroids to {OUTPUT}")
    for biome, info in centroids.items():
        print(f"{biome}: count={info['count']} hsv={info['hsv_mean']} bgr={info['bgr_mean']}")


if __name__ == '__main__':
    main()
