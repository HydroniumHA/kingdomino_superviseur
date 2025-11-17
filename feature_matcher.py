import cv2
import numpy as np
import os
import argparse

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


def list_images(dirpath):
    if not os.path.isdir(dirpath):
        return []
    return [os.path.join(dirpath, f) for f in sorted(os.listdir(dirpath)) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def create_detector(method='orb'):
    method = method.lower()
    if method == 'sift':
        if hasattr(cv2, 'SIFT_create'):
            return cv2.SIFT_create(), cv2.NORM_L2
        else:
            raise RuntimeError('SIFT not available in this OpenCV build. Use ORB instead.')
    # default ORB
    return cv2.ORB_create(nfeatures=1000), cv2.NORM_HAMMING


def match_images(img_path, candidates, method='orb', ratio=0.75):
    detector, norm = create_detector(method)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Input image not found: {img_path}')

    kp1, des1 = detector.detectAndCompute(img, None)
    if des1 is None or len(kp1) == 0:
        raise RuntimeError('No keypoints/descriptors found in input image')

    # BFMatcher with crossCheck False for ratio test
    bf = cv2.BFMatcher(norm)

    results = []
    for cand in candidates:
        img2 = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
        if img2 is None:
            continue
        kp2, des2 = detector.detectAndCompute(img2, None)
        if des2 is None or len(kp2) == 0:
            results.append((cand, 0, []))
            continue

        # KNN match and ratio test
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            # fallback single match
            matches = [[m] for m in bf.match(des1, des2)]

        good = []
        for m in matches:
            if len(m) == 2:
                m1, m2 = m
                if m1.distance < ratio * m2.distance:
                    good.append(m1)
            else:
                # single match
                if len(m) == 1:
                    good.append(m[0])

        results.append((cand, len(good), good, kp2))

    # sort by number of good matches desc
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted, (img, kp1, des1)


def draw_match(img1_kp_des, cand_path, cand_kp, good_matches, out_path=None):
    img1, kp1, des1 = img1_kp_des
    img2 = cv2.imread(cand_path)
    if img2 is None:
        return None
    # convert grayscale kp1-> but drawMatches expects color images; ensure both are color
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    draw = cv2.drawMatches(img1_color, kp1, img2, cand_kp, good_matches, None, flags=2)
    if out_path:
        cv2.imwrite(out_path, draw)
    return draw


def main():
    parser = argparse.ArgumentParser(description='Feature-match an image against a folder of clear images')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--dir', '-d', default='data/clear_images', help='Directory with candidate images')
    parser.add_argument('--method', choices=['orb', 'sift'], default='sift')
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--top', type=int, default=5, help='Number of top matches to report')
    parser.add_argument('--outdir', default=None, help='If set, save match visualization images to this folder')
    args = parser.parse_args()

    candidates = list_images(args.dir)
    if not candidates:
        print(f'No candidate images found in {args.dir}')
        return

    results, img1_kp_des = match_images(args.image, candidates, method=args.method, ratio=args.ratio)

    top_n = min(args.top, len(results))
    print(f'Found {len(results)} candidates; showing top {top_n}')
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    for i in range(top_n):
        cand, count, good, kp2 = results[i]
        print(f'{i+1}. {os.path.basename(cand)} - good matches: {count}')
        if args.outdir:
            out_path = os.path.join(args.outdir, f'match_{i+1}_{os.path.basename(cand)}')
            draw = draw_match(img1_kp_des, cand, kp2, good, out_path)
            if draw is not None:
                print(f'  saved visualization to {out_path}')


if __name__ == '__main__':
    main()
