import os
import argparse
from feature_matcher import match_images, list_images, draw_match
from classify_biome import load_samples, classify_against_samples, mean_hsv_of_image, infer_biome_name_from_filename
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Combine feature-matching and HSV sample matching')
    parser.add_argument('image', help='Path to input tile image')
    parser.add_argument('--clear_dir', '-c', default='data/clear_images', help='Directory with clear sample images for feature matching')
    parser.add_argument('--samples', '-s', default='biome_palette_samples.json', help='Samples JSON (filename->mean_hsv,mean_bgr)')
    parser.add_argument('--method', choices=['orb', 'sift'], default='sift', help='Feature detector to use')
    parser.add_argument('--ratio', type=float, default=0.75, help='Ratio for KNN matching')
    parser.add_argument('--match_threshold', '-t', type=int, default=15, help='Minimum number of good matches to accept feature-match result')
    parser.add_argument('--hsv_threshold', type=float, default=15, help='Maximum HSV distance to accept HSV match')
    parser.add_argument('--hsv_check_factor', type=float, default=1.5, help='Multiplier for HSV threshold when validating feature-match candidate')
    parser.add_argument('--top', type=int, default=5, help='How many top matches to consider/print')
    parser.add_argument('--outdir', default=None, help='If set, save visualization images for top matches')
    parser.add_argument('--lab_threshold', type=float, default=20.0, help='Maximum CIELab distance to accept BGR/Lab color match')
    args = parser.parse_args()

    # Load HSV samples early so we can use color-consistency checks
    try:
        samples = load_samples(args.samples)
    except Exception as e:
        print(f'Error loading samples file: {e}; proceeding without samples')
        samples = {}

    mean_hsv = mean_hsv_of_image(args.image)
    if mean_hsv is None:
        print('Error: failed to compute mean HSV of input image')
        return
    print(f'Input mean HSV: {[int(round(x)) for x in mean_hsv]}')
    # compute mean BGR as well for Lab distance checks
    img_color = cv2.imread(args.image)
    if img_color is None:
        print(f'Error: failed to load image for BGR mean: {args.image}')
        return
    hsv_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    v = hsv_img[:, :, 2]
    s = hsv_img[:, :, 1]
    mask = (v > 10) & (s > 10)
    if not np.any(mask):
        mask = np.ones_like(v, dtype=bool)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mean_bgr = cv2.mean(img_color, mask=mask_u8)[:3]
    mean_bgr = [float(x) for x in mean_bgr]
    print(f'Input mean BGR: {[int(round(x)) for x in mean_bgr]}')

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

    # Print top feature-match candidates for debugging
    if candidates and results:
        top_n = min(args.top, len(results))
        print(f'Top {top_n} feature-match candidates:')
        for i in range(top_n):
            cand, cnt, good, kp2 = results[i]
            print(f'  {i+1}. {os.path.basename(cand)} - good matches: {cnt}')
    else:
        print('No feature-match results')

    print(f'Feature-matching best: {os.path.basename(best_file) if best_file else None} matches={best_count}')

    # Compute HSV best match regardless (we'll print its distance)
    sample_best_file, sample_biome, sample_dist = classify_against_samples(mean_hsv, samples, mode='hsv')
    print(f'HSV best sample: {sample_best_file} -> biome {sample_biome} (HSV dist={sample_dist:.2f})')

    # Detailed sample distances: compute best per biome for diagnosis
    try:
        import numpy as _np
        biome_best = {}
        for fname, info in samples.items():
            biome = infer_biome_name_from_filename(fname)
            s_h = _np.array(info['mean_hsv'], dtype=float)
            d = float(_np.linalg.norm(_np.array(mean_hsv, dtype=float) - s_h))
            if biome not in biome_best or d < biome_best[biome][1]:
                biome_best[biome] = (fname, d)

        print('Best sample per biome:')
        for b, (f, d) in sorted(biome_best.items(), key=lambda x: x[1][1]):
            print(f'  {b}: {f} (dist={d:.2f})')
    except Exception:
        pass

    # Also compute best per biome using Lab (more perceptual) where possible
    try:
        biome_best_lab = {}
        inp_lab = cv2.cvtColor(np.uint8([[[int(round(mean_bgr[0])), int(round(mean_bgr[1])), int(round(mean_bgr[2]))]]]), cv2.COLOR_BGR2LAB)[0,0].astype(float)
        for fname, info in samples.items():
            biome = infer_biome_name_from_filename(fname)
            samp_bgr = info.get('mean_bgr', [0,0,0])
            samp_lab = cv2.cvtColor(np.uint8([[[int(round(samp_bgr[0])), int(round(samp_bgr[1])), int(round(samp_bgr[2]))]]]), cv2.COLOR_BGR2LAB)[0,0].astype(float)
            d_lab = float(np.linalg.norm(inp_lab - samp_lab))
            if biome not in biome_best_lab or d_lab < biome_best_lab[biome][1]:
                biome_best_lab[biome] = (fname, d_lab)

        print('Best sample per biome (Lab distances):')
        for b, (f, d) in sorted(biome_best_lab.items(), key=lambda x: x[1][1]):
            print(f'  {b}: {f} (Lab dist={d:.2f})')
    except Exception:
        pass

    # derive overall best biome by Lab distance (lowest Lab dist)
    sample_best_lab_file = None
    sample_best_lab_biome = None
    sample_best_lab_dist = None
    if 'biome_best_lab' in locals() and biome_best_lab:
        # find biome with smallest lab distance
        best_b, (best_fname, best_d) = min(biome_best_lab.items(), key=lambda x: x[1][1])
        sample_best_lab_file = best_fname
        sample_best_lab_biome = best_b
        sample_best_lab_dist = best_d
        print(f'Lab best sample: {sample_best_lab_file} -> biome {sample_best_lab_biome} (Lab dist={sample_best_lab_dist:.2f})')
    else:
        sample_best_lab_file = None
        sample_best_lab_biome = None
        sample_best_lab_dist = None

    chosen_biome = None
    chosen_by = None

    # Decision rule: prefer feature-match if enough matches; else if HSV distance <= threshold choose HSV; otherwise fallback to HSV
    if best_count and best_count >= args.match_threshold:
        # Before accepting feature-match, verify color consistency with HSV samples (if available)
        candidate_name = os.path.basename(best_file) if best_file else None
        candidate_biome = infer_biome_name_from_filename(candidate_name) if candidate_name else None
        chosen_biome = candidate_biome
        # compute HSV distance to the sample that matches this candidate filename (if present)
        hsv_consistent = True
        if candidate_name and candidate_name in samples:
            cand_mean = samples[candidate_name]['mean_hsv']
            import numpy as _np
            d_cand = float(_np.linalg.norm(_np.array(mean_hsv, dtype=float) - _np.array(cand_mean, dtype=float)))
            print(f'  Feature candidate HSV distance to sample {candidate_name}: {d_cand:.2f}')
            if d_cand > args.hsv_threshold * args.hsv_check_factor:
                hsv_consistent = False
                print(f'  Candidate rejected: HSV distance {d_cand:.2f} > {args.hsv_threshold * args.hsv_check_factor:.2f} (threshold*factor)')
            else:
                # also check Lab distance (more perceptual) using mean_bgr if available
                samp_bgr = samples[candidate_name].get('mean_bgr')
                if samp_bgr is not None:
                    inp_lab = cv2.cvtColor(np.uint8([[[int(round(mean_bgr[0])), int(round(mean_bgr[1])), int(round(mean_bgr[2]))]]]), cv2.COLOR_BGR2LAB)[0,0].astype(float)
                    samp_lab = cv2.cvtColor(np.uint8([[[int(round(samp_bgr[0])), int(round(samp_bgr[1])), int(round(samp_bgr[2]))]]]), cv2.COLOR_BGR2LAB)[0,0].astype(float)
                    d_lab_cand = float(np.linalg.norm(inp_lab - samp_lab))
                    print(f'  Feature candidate Lab distance to sample {candidate_name}: {d_lab_cand:.2f}')
                    if d_lab_cand > args.lab_threshold:
                        hsv_consistent = False
                        print(f'  Candidate rejected: Lab distance {d_lab_cand:.2f} > {args.lab_threshold:.2f}')

        if hsv_consistent:
            chosen_by = f'feature_match ({best_count} matches)'
            print(f'Accepted by feature matching -> biome: {chosen_biome}')
            if args.outdir and img1_kp_des is not None:
                os.makedirs(args.outdir, exist_ok=True)
                out_path = os.path.join(args.outdir, f'feature_match_top_{os.path.basename(best_file)}')
                draw = draw_match(img1_kp_des, best_file, best_kp2, best_good, out_path)
                if draw is not None:
                    print(f'  saved feature-match visualization to {out_path}')
        else:
            print('Falling back to HSV because feature-match candidate is color-inconsistent')
    else:
        # Use Lab-perceptual distances to choose color-match (preferred over raw HSV)
        if sample_best_lab_dist is not None and sample_best_lab_dist <= args.lab_threshold:
            chosen_biome = sample_best_lab_biome
            chosen_by = f'lab_sample (dist={sample_best_lab_dist:.2f})'
            print(f'Accepted by Lab color match (<= {args.lab_threshold}) -> biome: {chosen_biome}')
        else:
            # fallback to best Lab (even if > threshold) so we always pick something
            chosen_biome = sample_best_lab_biome or sample_biome
            chosen_by = f'lab_sample (dist={sample_best_lab_dist:.2f})' if sample_best_lab_dist is not None else f'hsv_sample (distance={sample_dist:.2f})'
            print(f'No strong feature match; Lab best -> {chosen_biome} (Lab dist={sample_best_lab_dist:.2f} if available, HSV best: {sample_biome} dist={sample_dist:.2f})')

    print('\nFinal decision:')
    print(f'  biome: {chosen_biome}')
    print(f'  chosen_by: {chosen_by}')


if __name__ == '__main__':
    main()
