import os
import argparse
from feature_matcher import match_images, list_images, draw_match
from classify_biome import load_samples, classify_against_samples, mean_hsv_of_image, infer_biome_name_from_filename


def main():
    parser = argparse.ArgumentParser(description='Combine feature-matching and HSV sample matching')
    parser.add_argument('image', help='Path to input tile image')
    parser.add_argument('--clear_dir', '-c', default='data/clear_images', help='Directory with clear sample images for feature matching')
    parser.add_argument('--samples', '-s', default='biome_palette_samples.json', help='Samples JSON (filename->mean_hsv,mean_bgr)')
    parser.add_argument('--method', choices=['orb', 'sift'], default='sift', help='Feature detector to use')
    parser.add_argument('--ratio', type=float, default=0.75, help='Ratio for KNN matching')
    parser.add_argument('--match_threshold', '-t', type=int, default=15, help='Minimum number of good matches to accept feature-match result')
    parser.add_argument('--hsv_threshold', type=float, default=15, help='Maximum HSV distance to accept HSV match')
    parser.add_argument('--top', type=int, default=5, help='How many top matches to consider/print')
    parser.add_argument('--outdir', default=None, help='If set, save visualization images for top matches')
    args = parser.parse_args()

    # 1) Try feature matching
    candidates = list_images(args.clear_dir)
    if not candidates:
        print(f'No candidate images found in {args.clear_dir}; falling back to HSV only')
        candidates = []

    best_file = None
    best_count = 0
    best_good = None
    best_kp2 = None
    img1_kp_des = None

    if candidates:
        try:
            results, img1_kp_des = match_images(args.image, candidates, method=args.method, ratio=args.ratio)
            if results:
                best_file, best_count, best_good, best_kp2 = results[0]
        except RuntimeError as e:
            print(f'Feature matching error or no keypoints: {e}; will fallback to HSV')
            results = []

    print(f'Feature-matching best: {os.path.basename(best_file) if best_file else None} matches={best_count}')

    # Compute HSV best match regardless (we'll print its distance)
    try:
        samples = load_samples(args.samples)
    except Exception as e:
        print(f'Error loading samples file: {e}')
        samples = {}

    mean_hsv = mean_hsv_of_image(args.image)
    if mean_hsv is None:
        print('Error: failed to compute mean HSV of input image')
        return

    sample_best_file, sample_biome, sample_dist = classify_against_samples(mean_hsv, samples, mode='hsv')
    print(f'HSV best sample: {sample_best_file} -> biome {sample_biome} (distance={sample_dist:.2f})')

    chosen_biome = None
    chosen_by = None

    # Decision rule: prefer feature-match if enough matches; else if HSV distance <= threshold choose HSV; otherwise fallback to HSV
    if best_count and best_count >= args.match_threshold:
        chosen_biome = infer_biome_name_from_filename(os.path.basename(best_file))
        chosen_by = f'feature_match ({best_count} matches)'
        print(f'Accepted by feature matching -> biome: {chosen_biome}')
        if args.outdir and img1_kp_des is not None:
            os.makedirs(args.outdir, exist_ok=True)
            out_path = os.path.join(args.outdir, f'feature_match_top_{os.path.basename(best_file)}')
            draw = draw_match(img1_kp_des, best_file, best_kp2, best_good, out_path)
            if draw is not None:
                print(f'  saved feature-match visualization to {out_path}')
    elif sample_dist <= args.hsv_threshold:
        chosen_biome = sample_biome
        chosen_by = f'hsv_sample (distance={sample_dist:.2f})'
        print(f'Accepted by HSV (distance <= {args.hsv_threshold}) -> biome: {chosen_biome}')
    else:
        # fallback: choose HSV best (user earlier preferred HSV fallback)
        chosen_biome = sample_biome
        chosen_by = f'hsv_sample (distance={sample_dist:.2f})'
        print(f'No strong feature match and HSV distance > {args.hsv_threshold}; fallback to HSV best -> {chosen_biome} (distance={sample_dist:.2f})')

    print('\nFinal decision:')
    print(f'  biome: {chosen_biome}')
    print(f'  chosen_by: {chosen_by}')


if __name__ == '__main__':
    main()
