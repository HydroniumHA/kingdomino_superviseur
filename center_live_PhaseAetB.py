#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kingdomino PHASE A + PHASE B 

# PHASE A: Lire et trier les chiffres (dominos)
# OCR : pr lire les dominos

# PHASE B: Vérifier l’ordre de jeu des pions (haut → milieu → bas)
# Pickup + drop : pr les pions

# Touches
#   ESC quitte
#   r repick ROI + reset tout
#   c reset phases A + B sans repick

import os
import time
import json

import cv2
import numpy as np
import easyocr

# ========== chemins ==========
VIDEO_SOURCE = "/Users/birjan/Documents/MASI/2025-2026/8.Visio/videos/test.mp4"
IP_SOURCE    = "http://192.168.2.147:8080/video"

USE_LIVE  = False
ROI_FILE  = "roi_config.json"
PICK_ROI  = False

# ========= perf et affichage =========
SCALE      = 0.90
STEP       = 3               # On saute des frames pr l OCR
TARGET_FPS = 24

# ========= ROI padding (agrandir la zone verte) =========
# -> augmenter valeurs pprour agrandir les rectangles verts (et la zone analysée)
ROI_PAD_X_FRAC = 0.12    # 0.00..0.30
ROI_PAD_Y_FRAC = 0.08    # 0.00..0.20

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

HUD_SHIFT_X = 220   # on pousse le texte vers la gauche
HUD_MIN_X   = 12    # on evite de sortir de l image

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

# ========= PHASE B mouvement pions =========
B_MOTION_THR = 6.0           # plus bas = plus sensible
B_CALM_TIME  = 0.8           # temps calme en col2 avant validation
B_PERSIST_MS = 1700          # bandeau ok / err
B_STEP       = 1             # on garde toutes les frames pr le mouvement

# ========= couleurs texte ordre =========
COL_OK  = (0, 255, 0)   # vert
COL_BAD = (0, 0, 255)   # rouge


# ========= persistance message ordre phase A =========
A_ORDER_PERSIST_S = 1.6   # durée d’affichage du message (secondes)

order_msg_text = None
order_msg_ok = True
order_msg_until = 0.0

def set_order_msg(txt, ok):
    global order_msg_text, order_msg_ok, order_msg_until
    order_msg_text = txt
    order_msg_ok = ok
    order_msg_until = time.time() + A_ORDER_PERSIST_S

def draw_order_msg_if_any(frame):
    global order_msg_text, order_msg_ok, order_msg_until
    if order_msg_text and time.time() < order_msg_until:
        put(order_msg_text, frame, (12, 80), 1.0, (COL_OK if order_msg_ok else COL_BAD), 2, (0,0,0))
    elif time.time() >= order_msg_until:
        order_msg_text = None


# ---------------------------------------------------------------
# AFFICHAGE : sert à superposer des infos sur la vidéo (texte, messages OK/erreur, état du programme) pr voir en direct ce que le programme détecte et décide.
# ---------------------------------------------------------------

def put(txt, img, org, scale=0.7, color=(255,255,255), thick=2, bg=(0,0,0)):
    x,y = org
    if bg is not None:
        (w,h),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img,(x-4,y-h-6),(x+w+4,y+6),bg,-1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_banner(img, text, ok=True, persist_alpha=0.88):
    H, W = img.shape[:2]
    color = (60,180,75) if ok else (40,40,200)
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (W,56), color, -1)
    img[:] = cv2.addWeighted(overlay, persist_alpha, img, 1 - persist_alpha, 0)
    put(text, img, (12,36), scale=1.15, color=(255,255,255), thick=2, bg=None)

# ---------------------------------------------------------------
# ROI HELPERS définir et gérer les zones d’intérêt (ROI) sur l'img (COL1 et COL2) : 
# je peux les sélectionner à la souris, les sauvegarder en coordonnées normalisées (indépendantes de la résolution), 
# les recharger, et agrandir un peu la zone (padding) pour être sûr de bien capturer les dominos/pions même si ça bouge un peu
# ---------------------------------------------------------------

def rect_to_norm(x,y,w,h,W,H):
    return [x/W, y/H, w/W, h/H]

def norm_to_rect(nx,ny,nw,nh,W,H):
    return int(nx*W), int(ny*H), int(nw*W), int(nh*H)

def expand_rect(x, y, w, h, W, H, pad_x_frac=0.0, pad_y_frac=0.0):
    px = int(round(w * pad_x_frac))
    py = int(round(h * pad_y_frac))
    x2 = max(0, x - px)
    y2 = max(0, y - py)
    w2 = min(W - x2, w + 2*px)
    h2 = min(H - y2, h + 2*py)
    return x2, y2, w2, h2

def pick_two_rois(first_frame, save_path):
    disp = first_frame.copy()
    put("Selectionne COL1 HAUT vers BAS", disp, (20,35))
    r1 = cv2.selectROI("ROI", disp, False, False)
    cv2.destroyWindow("ROI")

    disp = first_frame.copy()
    put("Selectionne COL2 HAUT vers BAS", disp, (20,35))
    r2 = cv2.selectROI("ROI", disp, False, False)
    cv2.destroyWindow("ROI")

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
    with open(path,"r") as f:
        rois = json.load(f)
    if "col1" not in rois or "col2" not in rois:
        raise KeyError("roi_config.json doit contenir col1 et col2")
    return rois

# ---------------------------------------------------------------
# OCR: il initialise EasyOCR une seule fois (pour éviter de le recréer à chaque frame), 
# il redimensionne des bounding boxes qd on travaille sur des imgs downscalées (scale_box), 
# et il calcule un IoU (taux de recouvrement entre 2 rectangles) pour vérifier si une détection 
# est au même endroit qu’avant et donc la considérer comme stable/verrouillée.
# ---------------------------------------------------------------

_reader = None
def get_reader():
    global _reader
    if _reader is None:
        print("Init EasyOCR CPU")
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def scale_box(box, sx, sy=None):
    if box is None:
        return None
    if sy is None:
        sy = sx
    x,y,w,h = box
    return (int(x*sx), int(y*sy), int(w*sx), int(h*sy))

def iou(a, b_):
    if a is None or b_ is None:
        return 0.0
    ax,ay,aw,ah = a
    bx,by,bw,bh = b_
    xA = max(ax, bx); yA = max(ay, by)
    xB = min(ax+aw, bx+bw); yB = min(ay+ah, by+bh)
    inter = max(0, xB-xA) * max(0, yB-yA)
    union = aw*ah + bw*bh - inter + 1e-9
    return inter/union

# ---------------------------------------------------------------
# Détection bbox domino: rouver automatiquement le rectangle (bbox) du domino dans une bande d’img : 
# il améliore le contraste (Lab + CLAHE), fait des traitements morphologiques + seuillage pour isoler la forme, 
# récupère les contours et garde le meilleur rectangle selon taille/forme attendues ; 
# si ça rate, il utilise un plan B avec détection de bords (Canny) puis renvoie la bbox du domino avec un petit padding.
# ---------------------------------------------------------------

def detect_domino_bbox(band_gray):
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

# ---------------------------------------------------------------
# bbox digits: localise les chiffres dans le domino puis les lit : d’abord find_digits_bbox() fait du traitement d'img (contraste + morphologie + contours) pr trouver la zone la plus probable des chiffres, 
# ensuite read_number_from_crop() découpe cette zone, l’agrandit et lance EasyOCR pr obtenir le nbr (avec un plan B _raw_ocr() qui tente l’OCR sur toute la tuile si on n’a pas réussi à isoler la zone des chiffres)
# ---------------------------------------------------------------

def find_digits_bbox(tile_gray):
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
# PHASE A : lire les 3 chiffres d’une colonne (haut/milieu/bas) de façon fiable et décider si l’ordre est correct, 
# puis détecter quand la colonne 2 n’affiche plus de chiffres.
# ---------------------------------------------------------------

def reset_state():
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
    return (vals[0] is not None and vals[1] is not None and vals[2] is not None and
            vals[0] <= vals[1] <= vals[2])

def _rect_from_norm_with_pad(col_norm, W, H):
    x, y, w, h = norm_to_rect(*col_norm, W, H)
    x, y, w, h = expand_rect(x, y, w, h, W, H, ROI_PAD_X_FRAC, ROI_PAD_Y_FRAC)
    return x, y, w, h

def read_column_sequential(gray, frame, col_norm, st, label, col_color, dom_color, global_freeze_until_ref):
    W, H = frame.shape[1], frame.shape[0]

    # ROI pixels + padding
    x1, y1, w1, h1 = _rect_from_norm_with_pad(col_norm, W, H)
    col = gray[y1:y1+h1, x1:x1+w1]
    ch, cw = col.shape[:2]
    rh = ch//3 if ch >= 3 else ch

    # rectangle colonne (vert/orange)
    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), col_color, 2)

    # HUD a gauche de la colonne
    hudx = max(HUD_MIN_X, x1 - HUD_SHIFT_X)

    KEEP_AFTER_MISS = 6

    for b in range(3):
        y0   = b*rh
        y1b  = (b+1)*rh if b<2 else ch
        band = col[y0:y1b, :]

        if b != st["target_idx"]:
            txt = ("no dom" if st["vals"][b] is None else str(st["vals"][b]))
            put(f"{label}[{b}] {txt}", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        if band.size == 0:
            put(f"{label}[{b}] no dom", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        band_small = cv2.resize(
            band,
            (max(1, int(band.shape[1]*DET_DOWNSCALE)), max(1, int(band.shape[0]*DET_DOWNSCALE))),
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
                put(f"{label}[{b}] no dom", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
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
            global_freeze_until_ref[0] = max(global_freeze_until_ref[0], time.time() + FREEZE_ON_LOCK_S)
            st["next_ocr_t"][b] = time.time()

            tile_gray_full = band[by:by+bh, bx:bx+bw]
            if tile_gray_full.size == 0:
                st["next_ocr_t"][b] = time.time() + 0.20
                put(f"{label}[{b}] vide", frame, (hudx, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                continue

            if cv2.Laplacian(tile_gray_full, cv2.CV_64F).var() < BLUR_VAR_T:
                st["next_ocr_t"][b] = time.time() + 0.20
                put(f"{label}[{b}] flou", frame, (hudx, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                continue

            small = cv2.resize(tile_gray_full, (160,160), interpolation=cv2.INTER_AREA)
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
                put(f"{label}[{b}] bouge", frame, (hudx, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                continue

            if SKIN_SUPPRESS:
                tile_bgr = frame[y1+y0+by:y1+y0+by+bh, x1+bx:x1+bx+bw]
                if tile_bgr.size > 0:
                    ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
                    lower = np.array([0,133,77], dtype=np.uint8)
                    upper = np.array([255,173,127], dtype=np.uint8)
                    mask = cv2.inRange(ycrcb, lower, upper)
                    if (mask > 0).mean() > SKIN_FRAC_MAX:
                        st["next_ocr_t"][b] = time.time() + 0.20
                        put(f"{label}[{b}] main", frame, (hudx, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                        continue

        cur_val = None
        now = time.time()
        if st["locked"][b] and (newly_locked or now >= st["next_ocr_t"][b]):
            tile_gray = band[by:by+bh, bx:bx+bw]
            if tile_gray.size > 0:
                cur_val, digits_box = read_number_from_crop(tile_gray)
            else:
                cur_val, digits_box = (None, None)
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
            frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))

def col2_has_any_digits(gray, frame, col2_norm, col_color, dom_color):
    W, H = frame.shape[1], frame.shape[0]

    x1, y1, w1, h1 = _rect_from_norm_with_pad(col2_norm, W, H)
    col = gray[y1:y1+h1, x1:x1+w1]
    ch, cw = col.shape[:2]
    rh = ch//3 if ch >= 3 else ch

    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), col_color, 2)

    hudx = max(HUD_MIN_X, x1 - HUD_SHIFT_X)

    any_digits = False
    for b in range(3):
        y0   = b*rh
        y1b  = (b+1)*rh if b<2 else ch
        band = col[y0:y1b, :]

        if band.size == 0:
            put(f"C2[{b}] no dom", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        band_small = cv2.resize(
            band,
            (max(1, int(band.shape[1]*DET_DOWNSCALE)), max(1, int(band.shape[0]*DET_DOWNSCALE))),
            interpolation=cv2.INTER_AREA
        )
        dom_box_small = detect_domino_bbox(band_small)
        dom_box = scale_box(dom_box_small, 1.0/DET_DOWNSCALE)

        if dom_box is None:
            put(f"C2[{b}] no dom", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
            continue

        bx,by,bw,bh = dom_box
        cv2.rectangle(frame, (x1+bx, y1+y0+by), (x1+bx+bw, y1+y0+by+bh), dom_color, 2)

        tile_gray = band[by:by+bh, bx:bx+bw]
        has_dig = (tile_gray.size > 0 and find_digits_bbox(tile_gray) is not None)

        if has_dig:
            any_digits = True
            put(f"C2[{b}] digits", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
        else:
            put(f"C2[{b}] no dig", frame, (hudx, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))

    return any_digits

# ---------------------------------------------------------------
# PHASE B : déterminer dans quel ordre les pions sont joués (HAUT → MILIEU → BAS) en détectant le mouvement dans COL1 (pickup) puis dans COL2 (drop), 
# et en validant seulement qd ça redevient calme.
#   - RowMotion : transforme une ROI (une colonne) en détecteur de mouvement par ligne (haut/milieu/bas) en comparant l’image actuelle à la précédente (diff), 
#     puis en sortant un score de mouvement pour chaque bande.
#   - TurnManager : fait la machine à états de la Phase B : il sait quel pion est attendu, vérifie que le bon pion bouge au bon moment 
#     (pickup en COL1 puis arrivée en COL2), affiche un message OK/ERREUR, et passe au pion suivant seulement après une période de calme (drop validé).
# ---------------------------------------------------------------

class RowMotion:
    def __init__(self, roi_rect, rows=3, thr=6.0):
        self.roi_rect = roi_rect
        self.rows = rows
        self.thr = float(thr)
        self.prev = None

    def reset_visual_memory(self):
        self.prev = None

    def step(self, frame_bgr):
        x,y,w,h = self.roi_rect
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            return None, None, None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        if self.prev is None:
            self.prev = gray
            return None, None, None

        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray

        rh = max(1, h // self.rows)
        scores = []
        for i in range(self.rows):
            sub = diff[i*rh:min((i+1)*rh, diff.shape[0]), :]
            scores.append(float(sub.mean()) if sub.size else 0.0)

        scores = np.array(scores, dtype=np.float32)
        idx_max = int(np.argmax(scores))
        val_max = float(scores[idx_max])
        return idx_max, val_max, scores

class TurnManager:
    def __init__(self, move_threshold=6.0, calm_time=0.8, banner_ms=1700):
        self.move_threshold = float(move_threshold)
        self.calm_time = float(calm_time)
        self.banner_s = (banner_ms / 1000.0)
        self.reset_all()

    def row_name(self, r):
        return "HAUT" if r == 0 else ("MILIEU" if r == 1 else "BAS")

    def hud_expected(self):
        if self.sequence_finie:
            return "PHASE B ok  Sequence finie HAUT -> MILIEU -> BAS"
        return f"PHASE B  Tour attendu {self.row_name(self.expected_row)}  state {self.state}"

    def _say_banner(self, txt, ok, now):
        self.last_banner_text = txt
        self.last_banner_ok = ok
        self.banner_until = now + self.banner_s

    def get_banner_to_draw(self, now):
        if now < self.banner_until and self.last_banner_text:
            return self.last_banner_text, self.last_banner_ok
        return None, None

    def _next_player(self):
        if self.expected_row < 2:
            self.expected_row += 1
            self.state = "waiting_pickup"
        else:
            self.sequence_finie = True
            self.state = "done"

    def reset_all(self):
        self.expected_row = 0
        self.sequence_finie = False
        self.state = "waiting_pickup"
        self.last_big_move_time_col2 = 0.0
        self.last_banner_text = None
        self.last_banner_ok = True
        self.banner_until = 0.0

    def update(self, now, scores1, scores2):
        if self.sequence_finie or self.state == "done":
            return

        cur_row = self.expected_row
        cur_name = self.row_name(cur_row)

        # 1 waiting_pickup
        if self.state == "waiting_pickup":
            if scores1 is not None:
                if scores1[cur_row] >= self.move_threshold:
                    self.state = "carrying"
                    self._say_banner(f"{cur_name} commence -> OK pickup", True, now)
                    return

                for r in [0,1,2]:
                    if r == cur_row:
                        continue
                    if r < cur_row:
                        continue
                    if scores1[r] >= self.move_threshold:
                        self._say_banner(
                            f"{self.row_name(r)} bouge alors que c est {cur_name} -> ERREUR",
                            False, now
                        )
                        return
            return

        # 2 carrying
        if self.state == "carrying":
            if scores2 is not None:
                if scores2[cur_row] >= self.move_threshold:
                    self.state = "waiting_drop"
                    self.last_big_move_time_col2 = now
                    self._say_banner(f"{cur_name} arrive en col2  check stabilité", True, now)
                    return
            return

        # 3 waiting_drop
        if self.state == "waiting_drop":
            if scores2 is not None:
                if scores2[cur_row] >= self.move_threshold:
                    self.last_big_move_time_col2 = now

            if (now - self.last_big_move_time_col2) >= self.calm_time:
                self._say_banner(f"{cur_name} VALIDÉ  joueur suivant", True, now)
                self._next_player()
            return

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    src = IP_SOURCE if USE_LIVE else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d’ouvrir la source: {src}")

    ok, f0 = cap.read()
    if not ok:
        raise RuntimeError("Premiere frame vide / impossible de lire")

    H0, W0 = f0.shape[:2]
    W, H = int(W0*SCALE), int(H0*SCALE)
    first = cv2.resize(f0, (W, H))

    rois = pick_two_rois(first, ROI_FILE) if PICK_ROI else load_rois(ROI_FILE)
    col1_norm = rois["col1"]
    col2_norm = rois["col2"]

    win = "Kingdomino PHASE A + B"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 980, int(980*H/W))

    # Etats Phase A
    st1 = reset_state()
    st2 = reset_state()

    PH_A1 = 0
    PH_A2 = 1
    PH_A3 = 2
    PH_B  = 3
    phase = PH_A1

    sort_ok_streak = 0
    no_digits_streak = 0

    # Etats Phase B
    motion_col1 = None
    motion_col2 = None
    tm = None

    # Cadence + freeze OCR
    i = 0
    tprev = time.time()
    raw_prev = None
    global_freeze_until = 0.0

    def reset_all_phases():
        nonlocal st1, st2, phase, sort_ok_streak, no_digits_streak
        nonlocal motion_col1, motion_col2, tm, global_freeze_until
        st1 = reset_state()
        st2 = reset_state()
        phase = PH_A1
        sort_ok_streak = 0
        no_digits_streak = 0
        motion_col1 = None
        motion_col2 = None
        tm = None
        global_freeze_until = 0.0

    while True:
        now_loop = time.time()

        # step selon phase
        step_eff = B_STEP if phase == PH_B else STEP

        # Freeze uniquement en phase A
        if phase != PH_B and now_loop < global_freeze_until and raw_prev is not None:
            raw = raw_prev
            ok = True
        else:
            ok, raw = cap.read()
            if not ok:
                break
            raw_prev = raw.copy()

        # Skip frames si on veut
        if step_eff > 1 and (i % step_eff) != 0:
            i += 1
            continue
        i += 1

        frame = cv2.resize(raw, (W, H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- PHASE A ---
        if phase == PH_A1:
            put("PHASE A1  On lit COL1 puis tri", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            global_freeze_ref = [global_freeze_until]
            read_column_sequential(gray, frame, col1_norm, st1, "C1", (0,255,0), (0,255,128), global_freeze_ref)
            global_freeze_until = global_freeze_ref[0]

            if all(v is not None for v in st1["vals"]):
                ok_sort = is_sorted_ascending(st1["vals"])

                # TEXTE COULEUR (VERT/ROUGE)
                set_order_msg("ORDRE CORRECT" if ok_sort else "ORDRE INCORRECT", ok_sort)



                # petit debug valeurs
                put(f"C1 vals={st1['vals']}", frame, (12, 112), 0.8,(255,255,255),2,(0,0,0))

                if ok_sort:
                    sort_ok_streak += 1
                else:
                    sort_ok_streak = 0
                    if RESET_ON_BAD_SORT:
                        st1 = reset_state()

                if sort_ok_streak >= SORT_STABILIZE_N:
                    sort_ok_streak = 0
                    phase = PH_A2
                    st2 = reset_state()

                draw_order_msg_if_any(frame)


        elif phase == PH_A2:
            put("PHASE A2  On lit COL2 puis tri", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            global_freeze_ref = [global_freeze_until]
            read_column_sequential(gray, frame, col2_norm, st2, "C2", (255,128,0), (255,180,0), global_freeze_ref)
            global_freeze_until = global_freeze_ref[0]

            if all(v is not None for v in st2["vals"]):
                ok_sort = is_sorted_ascending(st2["vals"])

                # TEXTE COULEUR (VERT/ROUGE)
                msg = "ORDRE CORRECT" if ok_sort else "ORDRE INCORRECT"
                put(msg, frame, (12, 80), 1.0, (COL_OK if ok_sort else COL_BAD), 2, (0,0,0))

                put(f"C2 vals={st2['vals']}", frame, (12, 112), 0.8,(255,255,255),2,(0,0,0))

                if ok_sort:
                    sort_ok_streak += 1
                else:
                    sort_ok_streak = 0
                    if RESET_ON_BAD_SORT:
                        st2 = reset_state()

                if sort_ok_streak >= SORT_STABILIZE_N:
                    sort_ok_streak = 0
                    phase = PH_A3
                    no_digits_streak = 0
                    global_freeze_until = 0.0

        elif phase == PH_A3:
            put("PHASE A3  On attend COL2 sans chiffres", frame, (12, 35), 0.75,(255,255,255),2,(0,0,0))
            any_digits = col2_has_any_digits(gray, frame, col2_norm, (255,128,0), (255,180,0))

            if any_digits:
                no_digits_streak = 0
                put("COL2 encore chiffres", frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))
            else:
                no_digits_streak += 1
                put(f"COL2 aucun chiffre {no_digits_streak}/{NO_DIGITS_STABLE_FRAMES}",
                    frame, (12, 80), 0.85,(255,255,255),2,(0,0,0))

            if no_digits_streak >= NO_DIGITS_STABLE_FRAMES:
                put("PHASE A OK  On passe en PHASE B pions", frame, (12, 120), 0.95,(255,255,255),2,(0,0,0))

                # Init PHASE B (ROI pixels + padding)
                x1,y1,w1,h1 = _rect_from_norm_with_pad(col1_norm, W, H)
                x2,y2,w2,h2 = _rect_from_norm_with_pad(col2_norm, W, H)

                motion_col1 = RowMotion((x1,y1,w1,h1), rows=3, thr=B_MOTION_THR)
                motion_col2 = RowMotion((x2,y2,w2,h2), rows=3, thr=B_MOTION_THR)
                tm = TurnManager(move_threshold=B_MOTION_THR, calm_time=B_CALM_TIME, banner_ms=B_PERSIST_MS)

                global_freeze_until = 0.0
                phase = PH_B

        # --- PHASE B ---
        else:
            put("PHASE B  On verif ordre pions pickup COL1 drop COL2", frame, (12, 35), 0.72,(255,255,255),2,(0,0,0))

            x1,y1,w1,h1 = _rect_from_norm_with_pad(col1_norm, W, H)
            x2,y2,w2,h2 = _rect_from_norm_with_pad(col2_norm, W, H)
            motion_col1.roi_rect = (x1,y1,w1,h1)
            motion_col2.roi_rect = (x2,y2,w2,h2)

            _, _, scores1 = motion_col1.step(frame)
            _, _, scores2 = motion_col2.step(frame)

            now = time.time()
            tm.update(now, scores1, scores2)

            cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)
            cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (255,0,0), 2)

            if scores1 is not None:
                put(f"col1 mv {scores1[0]:.1f} {scores1[1]:.1f} {scores1[2]:.1f}", frame, (12, 80), 0.6,(255,255,255),1,(0,0,0))
            if scores2 is not None:
                put(f"col2 mv {scores2[0]:.1f} {scores2[1]:.1f} {scores2[2]:.1f}", frame, (12, 102), 0.6,(255,255,255),1,(0,0,0))

            put(tm.hud_expected(), frame, (12, 132), 0.75,(255,255,255),2,(0,0,0))

            banner_text, banner_ok = tm.get_banner_to_draw(now)
            if banner_text:
                draw_banner(frame, banner_text, ok=banner_ok)

            if tm.sequence_finie:
                put("PHASE B OK  On a fini l ordre des pions", frame, (12, 160), 0.85,(255,255,255),2,(0,0,0))

        put(f"PH={phase}  C1={st1['vals']}  C2={st2['vals']}", frame, (12, 205), 0.70,(255,255,255),2,(0,0,0))

        nowf = time.time()
        fps = 1.0/max(1e-6, nowf-tprev); tprev = nowf
        put(f"fps~{int(fps)}", frame, (12, H-12), 0.8,(255,255,255),2,(0,0,0))

        cv2.imshow(win, frame)

        frame_dt = 1.0 / max(1, TARGET_FPS)
        elapsed = time.time() - nowf
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if k == ord('r'):
            rois = pick_two_rois(frame, ROI_FILE)
            col1_norm = rois["col1"]
            col2_norm = rois["col2"]
            reset_all_phases()

        if k == ord('c'):
            reset_all_phases()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
