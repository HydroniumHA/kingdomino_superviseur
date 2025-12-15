#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kingdomino PHASE A version nettoyée
# On garde la logique qui marche bien
# On lit 3 dominos dans COL1 puis COL2 en séquentiel haut milieu bas
# On valide une valeur après STABILIZE_N lectures identiques
# On vérifie tri croissant sur COL1 puis COL2
# On attend ensuite que COL2 ne montre plus de chiffres donc dominos retournés
# On garde le freeze + gating flou mouvement main pr éviter OCR foireux
# Touches
#   ESC quitte
#   r repick les 2 ROIs et reset
#   c reset phase A sans repick

import cv2, time, json, os, numpy as np
import easyocr

# ========== chemins ==========
VIDEO_SOURCE = "/Users/birjan/Documents/MASI/2025-2026/8.Visio/projetJeu/kingdomino/data/center/videos/plateau2_ok2_480p.mp4"
IP_SOURCE    = "http://192.168.2.147:8080/video"

USE_LIVE  = False
ROI_FILE  = "src/roi_config.json"
PICK_ROI  = False

# ========= perf et affichage =========
SCALE      = 0.90
STEP       = 3
TARGET_FPS = 24

# ========= détection domino =========
BAND_X_MARGIN_FRAC = 0.06
CLAHE_CLIP = 2.0
CLAHE_TILE = (3, 3)
TOPHAT_REL = 0.18
CLOSE_REL  = 0.08

DOM_AREA_MIN_FRAC = 0.06
DOM_AREA_MAX_FRAC = 0.95
DOM_ASPECT_MIN    = 0.50
DOM_ASPECT_MAX    = 5.0
DOM_PAD_FRAC      = 0.06

EDGE_AREA_MIN_FRAC = 0.05
EDGE_CLOSE_REL     = 0.12

# ========= OCR / stabilisation =========
MAX_OCR_W     = 160
MAX_OCR_H     = 220
UPSCALE_DIGIT = 2.5
STABILIZE_N   = 2
LOCK_N        = 1

OCR_INTERVAL_S   = 0.80
DET_DOWNSCALE    = 0.51
FREEZE_ON_LOCK_S = 0.8

# ========= digits bbox =========
DIG_AREA_MIN_FRAC = 0.008
DIG_AREA_MAX_FRAC = 0.60
DIG_ASPECT_MIN    = 0.15
DIG_ASPECT_MAX    = 4.0
MIN_HEIGHT_FRAC   = 0.10
MERGE_GAP_X_REL   = 0.30

# ========= gating anti OCR =========
MOTION_DIFF_T = 12.0
STILL_FRAMES_FOR_OCR = 3
BLUR_VAR_T = 80.0
SKIN_SUPPRESS = True
SKIN_FRAC_MAX = 0.25

# ========= PHASE A =========
SORT_STABILIZE_N = 2
NO_DIGITS_STABLE_FRAMES = 8
RESET_ON_BAD_SORT = True

# ---------------------------------------------------------------

def put(txt, img, org, scale=0.7, color=(255,255,255), thick=2, bg=(0,0,0)):
    # On dessine du texte lisible
    x,y = org
    if bg is not None:
        (w,h),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img,(x-4,y-h-6),(x+w+4,y+6),bg,-1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def rect_to_norm(x,y,w,h,W,H):
    # On stocke ROI en normalisé
    return [x/W, y/H, w/W, h/H]

def norm_to_rect(nx,ny,nw,nh,W,H):
    # On repasse ROI en pixels
    return int(nx*W), int(ny*H), int(nw*W), int(nh*H)

def pick_two_rois(first_frame, save_path):
    # On pick COL1 puis COL2
    disp = first_frame.copy(); put("Selectionne COL1 HAUT vers BAS", disp, (20,35))
    r1 = cv2.selectROI("ROI", disp, False, False); cv2.destroyWindow("ROI")
    disp = first_frame.copy(); put("Selectionne COL2 HAUT vers BAS", disp, (20,35))
    r2 = cv2.selectROI("ROI", disp, False, False); cv2.destroyWindow("ROI")

    H,W = first_frame.shape[:2]
    rois = {"col1": rect_to_norm(*r1, W, H), "col2": rect_to_norm(*r2, W, H)}
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(save_path,"w") as f:
        json.dump(rois, f, indent=2)
    print(f"[ROI] Saved {save_path}")
    return rois

def load_rois(path):
    # On load ROI
    with open(path,"r") as f:
        rois = json.load(f)
    if "col1" not in rois:
        raise KeyError("roi_config.json doit contenir col1")
    if "col2" not in rois:
        print("[WARN] col2 absente donc on doit repick")
    return rois

_reader = None
def get_reader():
    # On init easyocr une seule fois
    global _reader
    if _reader is None:
        print("Init EasyOCR CPU")
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def scale_box(box, sx, sy=None):
    # On remap bbox si détection faite en downscale
    if box is None:
        return None
    if sy is None:
        sy = sx
    x,y,w,h = box
    return (int(x*sx), int(y*sy), int(w*sx), int(h*sy))

def iou(a, b_):
    # On calc iou simple
    if a is None or b_ is None:
        return 0.0
    ax,ay,aw,ah = a
    bx,by,bw,bh = b_
    xA = max(ax, bx); yA = max(ay, by)
    xB = min(ax+aw, bx+bw); yB = min(ay+ah, by+bh)
    inter = max(0, xB-xA) * max(0, yB-yA)
    union = aw*ah + bw*bh - inter + 1e-9
    return inter/union

# ---------- détection bbox domino ----------
def detect_domino_bbox(band_gray):
    # On cherche le gros rect du domino
    H, W = band_gray.shape[:2]
    if H < 40 or W < 80:
        return None

    mx = int(W * BAND_X_MARGIN_FRAC)
    x_off = mx if mx*2 < W else 0
    work = band_gray[:, x_off:W-x_off] if x_off > 0 else band_gray
    HH, WW = work.shape[:2]

    work_bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2Lab)
    _,_,b = cv2.split(lab)
    b_inv = 255 - b

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    b_eq = clahe.apply(b_inv)

    ksize = int(max(3, round(min(HH, WW) * TOPHAT_REL)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    tophat = cv2.morphologyEx(b_eq, cv2.MORPH_TOPHAT, kernel)

    bw = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, -3)

    kclose = max(3, int(round(WW * CLOSE_REL)))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kclose, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hkernel, iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def pick_rect(candidates, W_, H_):
        if not candidates:
            return None
        area_min = DOM_AREA_MIN_FRAC * (W_ * H_)
        area_max = DOM_AREA_MAX_FRAC * (W_ * H_)
        best=None; bestA=-1
        for c in candidates:
            x,y,w,h = cv2.boundingRect(c)
            A=w*h
            if A < area_min or A > area_max:
                continue
            ar = w/float(h)
            if not (DOM_ASPECT_MIN <= ar <= DOM_ASPECT_MAX):
                continue
            if A > bestA:
                best=(x,y,w,h); bestA=A
        return best

    best = pick_rect(cnts, WW, HH)

    if best is None:
        edges = cv2.Canny(cv2.GaussianBlur(work,(3,3),0), 50, 130)
        kclose2 = max(3, int(round(WW * EDGE_CLOSE_REL)))
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kclose2, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker, iterations=1)
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
        cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts2:
            area_min = EDGE_AREA_MIN_FRAC * (WW * HH)
            big = [c for c in cnts2 if cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3] >= area_min]
            best = pick_rect(big, WW, HH)

    if best is None:
        return None

    x,y,w,h = best
    px = int(w*DOM_PAD_FRAC); py = int(h*DOM_PAD_FRAC)
    x = max(0, x-px); y = max(0, y-py)
    w = min(WW-1, w+2*px); h = min(HH-1, h+2*py)
    return (x + x_off, y, w, h)

# ---------- bbox digits ----------
def find_digits_bbox(tile_gray):
    # On tente d’isoler la zone digits
    H, W = tile_gray.shape[:2]
    if H < 30 or W < 30:
        return None

    bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    _,_,b = cv2.split(lab)
    b = 255 - b

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    b = clahe.apply(b)

    ksize = max(3, int(round(min(H, W) * TOPHAT_REL)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    tophat = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, kernel)

    bw = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, -5)
    bw = cv2.medianBlur(bw, 3)

    kclose = max(3, int(round(W * CLOSE_REL)))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kclose, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hkernel, iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    area_min = DIG_AREA_MIN_FRAC * (H * W)
    area_max = DIG_AREA_MAX_FRAC * (H * W)

    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        a = w*h
        if a < area_min or a > area_max:
            continue
        ar = w/float(h)
        if not (DIG_ASPECT_MIN <= ar <= DIG_ASPECT_MAX):
            continue
        if h < MIN_HEIGHT_FRAC * H:
            continue
        boxes.append((x,y,w,h))

    if not boxes:
        return None

    boxes.sort(key=lambda t: t[0])
    merged = [boxes[0]]
    max_gap = MERGE_GAP_X_REL * W

    for bx,by,bw_,bh in boxes[1:]:
        mx,my,mw,mh = merged[-1]
        gap = bx - (mx + mw)
        y_overlap = min(my + mh, by + bh) - max(my, by)
        if gap < max_gap and y_overlap > 0.45 * min(mh, bh):
            nx = min(mx, bx)
            ny = min(my, by)
            nw = max(mx+mw, bx+bw_) - nx
            nh = max(my+mh, by+bh) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append((bx,by,bw_,bh))

    merged.sort(key=lambda t: t[2]*t[3], reverse=True)
    x,y,w,h = merged[0]

    padx = int(0.06 * w)
    pady = int(0.08 * h)
    x = max(0, x - padx); y = max(0, y - pady)
    w = min(W - 1, w + 2*padx); h = min(H - 1, h + 2*pady)
    return (x,y,w,h)

def _raw_ocr(img_gray):
    # OCR fallback direct
    g = cv2.GaussianBlur(img_gray, (3,3), 0)
    up = cv2.resize(g, None, fx=UPSCALE_DIGIT, fy=UPSCALE_DIGIT, interpolation=cv2.INTER_CUBIC)
    reader = get_reader()

    best = None
    for inv in (False, True):
        bw = cv2.threshold(up if not inv else 255-up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        res = reader.readtext(bw, detail=1, paragraph=False, allowlist="0123456789")
        if not res:
            continue
        txt, conf = res[0][1], float(res[0][2])
        digits = "".join(ch for ch in str(txt) if ch.isdigit())
        if digits and len(digits) <= 2:
            cand = (digits, conf)
            if best is None or (len(cand[0]) > len(best[0]) or cand[1] > best[1]):
                best = cand

    return int(best[0]) if best else None

def read_number_from_crop(tile_gray):
    # On limite taille pr perf OCR
    h,w = tile_gray.shape[:2]
    scale = min(MAX_OCR_W/float(w), MAX_OCR_H/float(h), 1.0)
    if scale < 1.0:
        tile_gray = cv2.resize(tile_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    digits_box = find_digits_bbox(tile_gray)
    if digits_box is not None:
        dx,dy,dw,dh = digits_box
        crop = tile_gray[dy:dy+dh, dx:dx+dw]
        crop = cv2.resize(crop, None, fx=UPSCALE_DIGIT, fy=UPSCALE_DIGIT, interpolation=cv2.INTER_CUBIC)

        reader = get_reader()
        candidates = []
        for inv in (False, True):
            bw = cv2.threshold(crop if not inv else 255-crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            res = reader.readtext(bw, detail=1, paragraph=False, allowlist="0123456789")
            if res:
                text, conf = res[0][1], float(res[0][2])
                digits = "".join(ch for ch in str(text) if ch.isdigit())
                if digits:
                    candidates.append((digits, conf))
        if candidates:
            candidates.sort(key=lambda t: (len(t[0]), t[1]), reverse=True)
            best_txt = candidates[0][0][:2]
            return int(best_txt), (dx,dy,dw,dh)

    val = _raw_ocr(tile_gray)
    return (val, None) if val is not None else (None, None)

# ---------------------------------------------------------------

def reset_state():
    # On garde un état proche du code original
    return {
        "vals": [None, None, None],
        "target_idx": 0,
        "last_read": None,
        "stable_cnt": 0,
        "locked": [False, False, False],
        "lock_cnt": [0, 0, 0],
        "dom_box_saved": [None, None, None],
        "prev_tile": [None, None, None],
        "still_cnt": [0, 0, 0],
        "dom_miss": [0, 0, 0],
        "next_ocr_t": [0.0, 0.0, 0.0],
    }

def is_sorted_ascending(vals):
    # On check tri croissant
    return (vals[0] is not None and vals[1] is not None and vals[2] is not None and
            vals[0] <= vals[1] <= vals[2])

def read_column_sequential(gray, frame, col_norm, st, label, col_color, dom_color, global_freeze_until_ref):
    # On lit une colonne exactement comme ton code original
    # On renvoie newly_locked pr permettre freeze global
    W,H = frame.shape[1], frame.shape[0]
    x1,y1,w1,h1 = norm_to_rect(*col_norm, W,H)
    col = gray[y1:y1+h1, x1:x1+w1]
    ch,cw = col.shape[:2]
    rh = ch//3

    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), col_color, 2)

    KEEP_AFTER_MISS = 6
    newly_locked_any = False

    for b in range(3):
        y0   = b*rh
        y1b  = (b+1)*rh if b<2 else ch
        band = col[y0:y1b, :]

        if b != st["target_idx"]:
            txt = ("no domino" if st["vals"][b] is None else str(st["vals"][b]))
            put(f"{label}[{b}] {txt}", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        band_small = cv2.resize(
            band,
            (int(band.shape[1]*DET_DOWNSCALE), int(band.shape[0]*DET_DOWNSCALE)),
            interpolation=cv2.INTER_AREA
        )
        dom_box_small = detect_domino_bbox(band_small)
        dom_box = scale_box(dom_box_small, 1.0/DET_DOWNSCALE)

        if dom_box is None:
            if st["dom_box_saved"][b] is not None and st["dom_miss"][b] < KEEP_AFTER_MISS:
                st["dom_miss"][b] += 1
                dom_box = st["dom_box_saved"][b]
            else:
                st["lock_cnt"][b] = 0
                st["locked"][b] = False
                st["dom_box_saved"][b] = None
                st["dom_miss"][b] = 0
                put(f"{label}[{b}] no domino", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
                continue
        else:
            st["dom_miss"][b] = 0

        bx,by,bw,bh = dom_box
        cv2.rectangle(frame, (x1+bx, y1+y0+by), (x1+bx+bw, y1+y0+by+bh), dom_color, 2)

        newly_locked = False
        if iou(st["dom_box_saved"][b], dom_box) > 0.5:
            st["lock_cnt"][b] += 1
        else:
            st["dom_box_saved"][b] = dom_box
            st["lock_cnt"][b] = 1

        if st["lock_cnt"][b] >= LOCK_N:
            if not st["locked"][b]:
                newly_locked = True
            st["locked"][b] = True

        if newly_locked:
            newly_locked_any = True
            global_freeze_until_ref[0] = max(global_freeze_until_ref[0], time.time() + FREEZE_ON_LOCK_S)
            st["next_ocr_t"][b] = time.time()

            tile_gray_full = band[by:by+bh, bx:bx+bw]

            if cv2.Laplacian(tile_gray_full, cv2.CV_64F).var() < BLUR_VAR_T:
                st["next_ocr_t"][b] = time.time() + 0.20
                put(f"{label}[{b}] flou", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                continue

            small = cv2.resize(tile_gray_full, (160, 160), interpolation=cv2.INTER_AREA)
            if st["prev_tile"][b] is None:
                st["prev_tile"][b] = small.copy()
                st["still_cnt"][b] = 0
            else:
                m = float(np.mean(cv2.absdiff(st["prev_tile"][b], small)))
                st["prev_tile"][b] = small.copy()
                if m < MOTION_DIFF_T:
                    st["still_cnt"][b] += 1
                else:
                    st["still_cnt"][b] = 0

            if st["still_cnt"][b] < STILL_FRAMES_FOR_OCR:
                st["next_ocr_t"][b] = time.time() + 0.15
                put(f"{label}[{b}] bouge", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                continue

            if SKIN_SUPPRESS:
                tile_bgr = frame[y1+y0+by:y1+y0+by+bh, x1+bx:x1+bx+bw]
                if tile_bgr.size > 0:
                    ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
                    lower = np.array([0, 133, 77], dtype=np.uint8)
                    upper = np.array([255, 173, 127], dtype=np.uint8)
                    mask = cv2.inRange(ycrcb, lower, upper)
                    if (mask > 0).mean() > SKIN_FRAC_MAX:
                        st["next_ocr_t"][b] = time.time() + 0.20
                        put(f"{label}[{b}] main", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                        continue

        cur_val = None
        now = time.time()
        if st["locked"][b] and (newly_locked or now >= st["next_ocr_t"][b]):
            tile_gray = band[by:by+bh, bx:bx+bw]
            cur_val, digits_box = read_number_from_crop(tile_gray)
            st["next_ocr_t"][b] = now + OCR_INTERVAL_S
            if digits_box is not None:
                dx,dy,dw,dh = digits_box
                cv2.rectangle(frame, (x1+bx+dx, y1+y0+by+dy),
                              (x1+bx+dx+dw, y1+y0+by+dy+dh), (0,255,0), 2)

        if cur_val is not None:
            if st["last_read"] == cur_val:
                st["stable_cnt"] += 1
            else:
                st["last_read"]  = cur_val
                st["stable_cnt"] = 1
            if st["stable_cnt"] >= STABILIZE_N:
                st["vals"][b] = cur_val
                st["target_idx"] = min(2, st["target_idx"]+1)
                st["last_read"]  = None
                st["stable_cnt"] = 0

        put(f"{label}[{b}] {('attend' if st['vals'][b] is None else st['vals'][b])}",
            frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))

    return newly_locked_any

def col2_has_any_digits(gray, frame, col2_norm, col_color, dom_color):
    # On check juste si on voit encore des digits sur les 3 dominos de COL2
    W,H = frame.shape[1], frame.shape[0]
    x1,y1,w1,h1 = norm_to_rect(*col2_norm, W,H)
    col = gray[y1:y1+h1, x1:x1+w1]
    ch,cw = col.shape[:2]
    rh = ch//3

    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), col_color, 2)

    any_digits = False
    for b in range(3):
        y0   = b*rh
        y1b  = (b+1)*rh if b<2 else ch
        band = col[y0:y1b, :]

        band_small = cv2.resize(
            band,
            (int(band.shape[1]*DET_DOWNSCALE), int(band.shape[0]*DET_DOWNSCALE)),
            interpolation=cv2.INTER_AREA
        )
        dom_box_small = detect_domino_bbox(band_small)
        dom_box = scale_box(dom_box_small, 1.0/DET_DOWNSCALE)

        if dom_box is None:
            put(f"C2[{b}] no dom", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        bx,by,bw,bh = dom_box
        cv2.rectangle(frame, (x1+bx, y1+y0+by), (x1+bx+bw, y1+y0+by+bh), dom_color, 2)

        tile_gray = band[by:by+bh, bx:bx+bw]
        has_dig = (tile_gray.size > 0 and find_digits_bbox(tile_gray) is not None)

        if has_dig:
            any_digits = True
            put(f"C2[{b}] digits", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
        else:
            put(f"C2[{b}] no dig", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))

    return any_digits

# ---------------------------------------------------------------

def main():
    # On ouvre la source
    src = IP_SOURCE if USE_LIVE else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)

    ok, f0 = cap.read()
    if not ok:
        raise RuntimeError("Impossible d’ouvrir la source")

    H0,W0 = f0.shape[:2]
    W,H = int(W0*SCALE), int(H0*SCALE)
    first = cv2.resize(f0,(W,H))

    rois = pick_two_rois(first, ROI_FILE) if PICK_ROI else load_rois(ROI_FILE)
    col1_norm = rois["col1"]
    col2_norm = rois.get("col2", None)
    if col2_norm is None:
        raise RuntimeError("col2 manquante dans roi_config.json donc on met PICK_ROI=True")

    win = "Kingdomino PHASE A clean"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 980, int(980*H/W))

    st1 = reset_state()
    st2 = reset_state()

    PH_COL1 = 0
    PH_COL2 = 1
    PH_WAIT = 2
    phase = PH_COL1

    sort_ok_streak = 0
    no_digits_streak = 0

    i = 0
    tprev = time.time()
    raw_prev = None
    global_freeze_until = 0.0

    while True:
        now_loop = time.time()

        if now_loop < global_freeze_until and raw_prev is not None:
            raw = raw_prev
        else:
            ok, raw = cap.read()
            if not ok:
                break
            raw_prev = raw.copy()

            if STEP > 1 and (i % STEP) != 0:
                i += 1
                continue
            i += 1

        frame = cv2.resize(raw,(W,H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        global_freeze_ref = [global_freeze_until]

        if phase == PH_COL1:
            put("PHASE A1  On lit COL1 puis tri", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            read_column_sequential(gray, frame, col1_norm, st1, "C1", (0,255,0), (0,255,128), global_freeze_ref)

            if st1["vals"][0] is not None and st1["vals"][1] is not None and st1["vals"][2] is not None:
                ok_sort = is_sorted_ascending(st1["vals"])
                put(f"C1 vals={st1['vals']} sorted={ok_sort}", frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))

                if ok_sort:
                    sort_ok_streak += 1
                else:
                    sort_ok_streak = 0
                    if RESET_ON_BAD_SORT:
                        st1 = reset_state()

                if sort_ok_streak >= SORT_STABILIZE_N:
                    sort_ok_streak = 0
                    phase = PH_COL2
                    st2 = reset_state()

        elif phase == PH_COL2:
            put("PHASE A2  On lit COL2 puis tri", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            read_column_sequential(gray, frame, col2_norm, st2, "C2", (255,128,0), (255,180,0), global_freeze_ref)

            if st2["vals"][0] is not None and st2["vals"][1] is not None and st2["vals"][2] is not None:
                ok_sort = is_sorted_ascending(st2["vals"])
                put(f"C2 vals={st2['vals']} sorted={ok_sort}", frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))

                if ok_sort:
                    sort_ok_streak += 1
                else:
                    sort_ok_streak = 0
                    if RESET_ON_BAD_SORT:
                        st2 = reset_state()

                if sort_ok_streak >= SORT_STABILIZE_N:
                    sort_ok_streak = 0
                    phase = PH_WAIT
                    no_digits_streak = 0

        else:
            put("PHASE A3  On attend COL2 sans chiffres", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            any_digits = col2_has_any_digits(gray, frame, col2_norm, (255,128,0), (255,180,0))

            if any_digits:
                no_digits_streak = 0
                put("COL2 encore chiffres", frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))
            else:
                no_digits_streak += 1
                put(f"COL2 aucun chiffre {no_digits_streak}/{NO_DIGITS_STABLE_FRAMES}", frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))

            if no_digits_streak >= NO_DIGITS_STABLE_FRAMES:
                put("PHASE A OK  On peut passer au round", frame, (12, 120), 0.95,(255,255,255),2,(0,0,0))

        global_freeze_until = global_freeze_ref[0]

        put(f"PH={phase}  C1={st1['vals']}  C2={st2['vals']}", frame, (12, 115), 0.75,(255,255,255),2,(0,0,0))
        now = time.time()
        fps = 1.0/max(1e-6, now-tprev); tprev = now
        put(f"fps~{int(fps)}", frame, (12, H-12), 0.8,(255,255,255),2,(0,0,0))

        cv2.imshow(win, frame)

        frame_dt = 1.0 / max(1, TARGET_FPS)
        elapsed = time.time() - now
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if k == ord('r'):
            rois = pick_two_rois(frame, ROI_FILE)
            col1_norm = rois["col1"]
            col2_norm = rois.get("col2", None)
            st1 = reset_state()
            st2 = reset_state()
            phase = PH_COL1
            sort_ok_streak = 0
            no_digits_streak = 0

        if k == ord('c'):
            st1 = reset_state()
            st2 = reset_state()
            phase = PH_COL1
            sort_ok_streak = 0
            no_digits_streak = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
