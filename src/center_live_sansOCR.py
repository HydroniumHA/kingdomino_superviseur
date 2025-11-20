#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kingdomino vidéo optimisée
# on détecte le domino, on rogne autour, on réduit la taille, on lit les chiffres par OCR
# on valide d’abord la tuile du haut puis celle du milieu puis celle du bas
# dépendances attendues opencv numpy easyocr

import cv2, time, json, os, numpy as np
import easyocr
from pathlib import Path

# ========== chemins ==========
VIDEO_SOURCE = "/Users/birjan/Documents/MASI/2025-2026/8.Visio/projetJeu/kingdomino/data/center/videos/plateau2_ok2_480p.mp4"  # on lit cette vidéo

# Quand on filme en live avec le GSM (flux IP)
# IP_SOURCE = "http://192.168.2.147:8080/video"

USE_LIVE   = False   # True = webcam, False = vidéo
#CAM_INDEX  = "http://192.168.2.147:8080/video"

ROI_FILE     = "src/roi_config.json"  # on garde ici la zone colonne
PICK_ROI     = False  # on remet à True une seule fois si on veut redessiner la zone colonne

# ========= paramètres perf et affichage =========
SCALE        = 0.90  # on réduit un peu l’image pour gagner des fps
STEP         = 3     # on ne traite qu’une frame sur X pour soulager le cpu
SHOW_DEBUG   = False # on affiche des vues intermédiaires si besoin

# ========= paramètres détection du domino par bande =========
BAND_X_MARGIN_FRAC = 0.06  # on coupe un peu les bords horizontaux pour éviter le décor
CLAHE_CLIP   = 2.0        # on renforce localement le contraste
CLAHE_TILE   = (3, 3)     # taille de tuile pour le clahe
TOPHAT_REL   = 0.18       # on retire le fond large avec un tophat
CLOSE_REL    = 0.08       # on bouche les petits trous en horizontal

# critères du grand rectangle du domino
DOM_AREA_MIN_FRAC = 0.06  # on ignore les trucs trop petits
DOM_AREA_MAX_FRAC = 0.95  # on ignore les trucs trop grands
DOM_ASPECT_MIN    = 0.50  # ratio largeur sur hauteur minimal accepté
DOM_ASPECT_MAX    = 5.0   # ratio maximal accepté
DOM_PAD_FRAC      = 0.06  # on élargit un peu la boîte du domino

# fallback bords si l’apparence échoue
EDGE_AREA_MIN_FRAC = 0.05 # on garde les contours assez grands
EDGE_CLOSE_REL     = 0.12 # fermeture pour relier les segments de bord

# ========= paramètres ocr et séquence =========
MAX_OCR_W     = 160   # on limite la largeur envoyée à l’ocr
MAX_OCR_H     = 220   # on limite la hauteur envoyée à l’ocr
OCR_EVERY_N   = 2     # on tente l’ocr plus souvent
UPSCALE_DIGIT = 2.5   # on agrandit les chiffres avant l’ocr
STABILIZE_N   = 2     # on valide après deux mêmes lectures
LOCK_N        = 1     # on verrouille la boîte du domino très vite

# ========= paramètres chiffres à l’intérieur du domino =========
DIG_AREA_MIN_FRAC = 0.008 # on accepte de petits blobs de chiffres
DIG_AREA_MAX_FRAC = 0.60  # on coupe si la boîte de chiffres est trop grande
DIG_ASPECT_MIN    = 0.15  # ratio minimal des chiffres
DIG_ASPECT_MAX    = 4.0   # ratio maximal des chiffres
MIN_HEIGHT_FRAC   = 0.10  # hauteur minimale du chiffre par rapport au domino
MERGE_GAP_X_REL   = 0.30  # on fusionne deux glyphes proches pour former 20 ou 43

# --- cadences et voie rapide ---
OCR_INTERVAL_S = 0.80  # on ne fait pas plus de 5 ocr par seconde et par tuile
DET_DOWNSCALE  = 0.51   # on détecte à 60 pourcent de la taille puis on remappe la boîte

# --- Anti-OCR quand ce n'est pas "propre" ---
MOTION_DIFF_T = 12.0         # seuil de mouvement moyen (0–255)
STILL_FRAMES_FOR_OCR = 3     # nb de frames stables avant OCR
BLUR_VAR_T = 80.0            # seuil de netteté (variance du Laplacien)
SKIN_SUPPRESS = True         # coupe l'OCR si on détecte beaucoup de peau
SKIN_FRAC_MAX = 0.25         # max 25 % de peau dans le crop

# --- Lecture vidéo & rythme ---
TARGET_FPS        = 24         # on n’affiche pas plus vite que 24 fps
FREEZE_ON_LOCK_S  = 0.8        # on fige ~0.8 s quand une tuile vient d’être verrouillée

# ---------------------------------------------------------------

def put(txt, img, org, scale=0.7, color=(255,255,255), thick=2, bg=(0,0,0)):
    # on dessine un texte lisible avec un fond noir
    x,y = org
    if bg is not None:
        (w,h),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img,(x-4,y-h-6),(x+w+4,y+6),bg,-1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def rect_to_norm(x,y,w,h,W,H): 
    # on passe une boîte en coordonnées normalisées
    return [x/W, y/H, w/W, h/H]

def norm_to_rect(nx,ny,nw,nh,W,H):
    # on repasse d’une boîte normalisée à des pixels
    return int(nx*W), int(ny*H), int(nw*W), int(nh*H)

def pick_two_rois(first_frame, save_path):
    # on choisit la colonne verticale une fois et on sauvegarde
    disp = first_frame.copy(); put("Selectionne la COLONNE 1 HAUT vers BAS", disp, (20,35))
    r1 = cv2.selectROI("ROI", disp, False, False); cv2.destroyWindow("ROI")
    disp = first_frame.copy(); put("(optionnel) COLONNE 2 ignore ici", disp, (20,35))
    r2 = cv2.selectROI("ROI", disp, False, False); cv2.destroyWindow("ROI")
    H,W = first_frame.shape[:2]
    rois = {"col1": rect_to_norm(*r1, W, H), "col2": rect_to_norm(*r2, W, H)}
    d = os.path.dirname(save_path)
    os.makedirs(d, exist_ok=True) if d else None
    with open(save_path,"w") as f: json.dump(rois, f, indent=2)
    print(f"[ROI] Saved to {save_path}")
    return rois

def load_rois(path):
    # on recharge la colonne déjà dessinée
    with open(path,"r") as f: 
        rois = json.load(f)
    if "col1" not in rois:
        raise KeyError("roi_config.json doit contenir col1")
    return rois

_reader = None
def get_reader():
    # on initialise easyocr une seule fois
    global _reader
    if _reader is None:
        print("Init EasyOCR CPU…")
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def safe_crop(img, x,y,w,h):
    # on recadre en restant dans l’image
    H,W = img.shape[:2]
    x0=max(0,x); y0=max(0,y)
    x1=min(W, x+w); y1=min(H, y+h)
    if x1<=x0 or y1<=y0: 
        return None
    return img[y0:y1, x0:x1]

def scale_box(box, sx, sy=None):
    # on remet une boîte à l’échelle souhaitée
    if box is None: 
        return None
    if sy is None: 
        sy = sx
    x,y,w,h = box
    return (int(x*sx), int(y*sy), int(w*sx), int(h*sy))

# ---------- détection du rectangle du domino dans une bande ----------
def detect_domino_bbox(band_gray):
    # on cherche un grand rectangle qui ressemble au domino
    H, W = band_gray.shape[:2]
    if H < 40 or W < 80:
        return None

    # on enlève un peu les bords latéraux de la bande
    mx = int(W * BAND_X_MARGIN_FRAC)
    x_off = mx if mx*2 < W else 0
    work = band_gray[:, x_off:W-x_off] if x_off > 0 else band_gray
    HH, WW = work.shape[:2]

    # voie apparence avec Lab b inversé tophat et fermeture
    work_bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2Lab)
    _,_,b = cv2.split(lab)
    b_inv = 255 - b
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    b_eq = clahe.apply(b_inv)
    ksize = int(max(3, round(min(HH, WW) * TOPHAT_REL)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    tophat = cv2.morphologyEx(b_eq, cv2.MORPH_TOPHAT, kernel)
    bw = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -3)
    kclose = max(3, int(round(WW * CLOSE_REL)))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kclose, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hkernel, iterations=1)

    # on récupère les contours candidats
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def pick_rect(candidates, W_, H_):
        # on garde le meilleur rectangle dans les bons ratios et les bonnes tailles
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

    # fallback bords si l’apparence ne marche pas
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

    # on ajoute un petit padding et on remet l’offset
    x,y,w,h = best
    px = int(w*DOM_PAD_FRAC); py = int(h*DOM_PAD_FRAC)
    x = max(0, x-px); y = max(0, y-py)
    w = min(WW-1, w+2*px); h = min(HH-1, h+2*py)
    return (x + x_off, y, w, h)

# ---------- détection de la boîte des chiffres dans le domino ----------
def find_digits_bbox(tile_gray):
    # on isole la zone où il y a les chiffres dans le domino recadré
    H, W = tile_gray.shape[:2]
    if H < 30 or W < 30:
        return None

    # on renforce les chiffres blancs par rapport au bois
    bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    _, _, b = cv2.split(lab)
    b = 255 - b
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    b = clahe.apply(b)

    # on nettoie puis on ferme horizontalement
    ksize = max(3, int(round(min(H, W) * TOPHAT_REL)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    tophat = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, kernel)
    bw = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5)
    bw = cv2.medianBlur(bw, 3)
    kclose = max(3, int(round(W * CLOSE_REL)))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kclose, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hkernel, iterations=1)

    # on prend les composants connexes candidats
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # on filtre par taille et ratio puis on fusionne les glyphes proches
    area_min = DIG_AREA_MIN_FRAC * (H * W)
    area_max = DIG_AREA_MAX_FRAC * (H * W)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h
        if a < area_min or a > area_max:
            continue
        ar = w / float(h)
        if not (DIG_ASPECT_MIN <= ar <= DIG_ASPECT_MAX):
            continue
        if h < MIN_HEIGHT_FRAC * H:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return None

    boxes.sort(key=lambda t: t[0])
    merged = [boxes[0]]
    max_gap = MERGE_GAP_X_REL * W
    for bx, by, bw_, bh in boxes[1:]:
        mx, my, mw, mh = merged[-1]
        gap = bx - (mx + mw)
        y_overlap = min(my + mh, by + bh) - max(my, by)
        if gap < max_gap and y_overlap > 0.45 * min(mh, bh):
            nx = min(mx, bx)
            ny = min(my, by)
            nw = max(mx + mw, bx + bw_) - nx
            nh = max(my + mh, by + bh) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append((bx, by, bw_, bh))

    merged.sort(key=lambda t: t[2] * t[3], reverse=True)
    x, y, w, h = merged[0]

    # on ajoute un léger padding pour ne pas couper la queue du 2
    padx = int(0.06 * w)
    pady = int(0.08 * h)
    x = max(0, x - padx); y = max(0, y - pady)
    w = min(W - 1, w + 2 * padx); h = min(H - 1, h + 2 * pady)
    return (x, y, w, h)

def prep_for_ocr(img_gray, invert=False):
    # on lisse et on binarise avant l’ocr
    g = cv2.GaussianBlur(img_gray,(3,3),0)
    _, bw = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if invert: 
        bw = 255 - bw
    return bw

def _raw_ocr(img_gray):
    # fallback ocr direct sur le domino entier quand la boîte de chiffres échoue
    g = cv2.GaussianBlur(img_gray, (3,3), 0)
    up = cv2.resize(g, None, fx=UPSCALE_DIGIT, fy=UPSCALE_DIGIT, interpolation=cv2.INTER_CUBIC)
    best = None
    reader = get_reader()
    for inv in (False, True):
        bw = cv2.threshold(up if not inv else 255 - up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        res = reader.readtext(bw, detail=1, paragraph=False, allowlist="0123456789")
        if not res:
            continue
        txt, conf = res[0][1], float(res[0][2])
        digits = "".join(ch for ch in str(txt) if ch.isdigit())
        if not digits:
            continue
        if len(digits) <= 2:
            cand = (digits, conf)
            if best is None or (len(cand[0]) > len(best[0]) or cand[1] > best[1]):
                best = cand
    return int(best[0]) if best else None

def read_number_from_crop(tile_gray):
    # on réduit la taille pour l’ocr si c’est trop grand
    h,w = tile_gray.shape[:2]
    scale = min(MAX_OCR_W/float(w), MAX_OCR_H/float(h), 1.0)
    if scale < 1.0:
        tile_gray = cv2.resize(tile_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # on essaie d’abord avec la boîte des chiffres
    digits_box = find_digits_bbox(tile_gray)
    if digits_box is not None:
        dx,dy,dw,dh = digits_box
        crop = tile_gray[dy:dy+dh, dx:dx+dw]
        crop = cv2.resize(crop, None, fx=UPSCALE_DIGIT, fy=UPSCALE_DIGIT, interpolation=cv2.INTER_CUBIC)
        reader = get_reader()
        candidates = []
        for inv in (False, True):
            bw = cv2.threshold(crop if not inv else 255 - crop, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            res = reader.readtext(bw, detail=1, paragraph=False, allowlist="0123456789")
            if res:
                text, conf = res[0][1], float(res[0][2])
                digits = "".join(ch for ch in str(text) if ch.isdigit())
                if digits:
                    candidates.append((digits, conf))
        if candidates:
            candidates.sort(key=lambda t: (len(t[0]), t[1]), reverse=True)
            best_txt = candidates[0][0][:2]  # on garde au plus deux chiffres
            return int(best_txt), (dx, dy, dw, dh)

    # sinon on tente l’ocr direct
    val = _raw_ocr(tile_gray)
    return (val, None) if val is not None else (None, None)

# ---------------------------------------------------------------

def main():
    # on ouvre la vidéo
    src = CAM_INDEX if USE_LIVE else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)

    #cap = cv2.VideoCapture(0 if str(VIDEO_SOURCE).isdigit() else VIDEO_SOURCE)
    ok, f0 = cap.read()
    if not ok:
       raise RuntimeError("Impossible d’ouvrir la source vidéo")

    # on prépare les tailles d’affichage
    H0,W0 = f0.shape[:2]
    W,H = int(W0*SCALE), int(H0*SCALE)
    first = cv2.resize(f0,(W,H))

    # on récupère la zone colonne
    rois = pick_two_rois(first, ROI_FILE) if PICK_ROI else load_rois(ROI_FILE)
    col1_norm = rois["col1"]

    # on ouvre la fenêtre vidéo
    win = "Kingdomino domino crop downscale OCR video"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 980, int(980*H/W))

    # on initialise l’état de lecture séquentielle
    vals         = [None, None, None]
    target_idx   = 0
    last_read    = None
    stable_cnt   = 0
    locked       = [False, False, False]
    lock_cnt     = [0, 0, 0]
    dom_box_saved= [None, None, None]
    prev_tile = [None, None, None]
    still_cnt = [0, 0, 0]


    # on gère la cadence et la tolérance de perte de boîte
    i = 0
    tprev = time.time()
    dom_miss = [0, 0, 0]
    KEEP_AFTER_MISS = 6
    next_ocr_t = [0.0, 0.0, 0.0]  # horloge par bande pour cadencer l’ocr
    raw_prev = None               # on garde la dernière frame pour pouvoir geler
    global_freeze_until = 0.0     # si > now, on réutilise la frame précédente



    while True:
        now_loop = time.time()
        
        # si on est en phase de freeze, on réutilise la dernière frame pour laisser l’OCR bosser
        if now_loop < global_freeze_until and raw_prev is not None:
            raw = raw_prev
        else:
            ok, raw = cap.read()
            
            if not ok:
                break
            raw_prev = raw.copy()
    
            # on applique STEP seulement quand on ne freeze pas
            if STEP > 1 and (i % STEP) != 0:
                i += 1
                continue
            i += 1


        # on prépare la frame
        frame = cv2.resize(raw,(W,H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # on extrait la colonne et on la coupe en trois bandes
        x1,y1,w1,h1 = norm_to_rect(*col1_norm, W,H)
        col = gray[y1:y1+h1, x1:x1+w1]
        ch,cw = col.shape[:2]
        rh = ch//3
        cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)

        for b in range(3):
            y0   = b*rh
            y1b  = (b+1)*rh if b<2 else ch
            band = col[y0:y1b, :]

            # on avance seulement quand la tuile précédente est validée
            if b != target_idx:
                txt = ("no domino" if vals[b] is None else str(vals[b]))
                put(f"[{b}] {txt}", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
                continue

            # détection du domino en petit puis remap à la taille réelle
            band_small = cv2.resize(
                band,
                (int(band.shape[1]*DET_DOWNSCALE), int(band.shape[0]*DET_DOWNSCALE)),
                interpolation=cv2.INTER_AREA
            )
            dom_box_small = detect_domino_bbox(band_small)
            dom_box = scale_box(dom_box_small, 1.0/DET_DOWNSCALE)

            # si on perd la boîte un instant on garde l’ancienne quelques frames
            if dom_box is None:
                if dom_box_saved[b] is not None and dom_miss[b] < KEEP_AFTER_MISS:
                    dom_miss[b] += 1
                    dom_box = dom_box_saved[b]
                else:
                    lock_cnt[b] = 0
                    locked[b] = False
                    dom_box_saved[b] = None
                    dom_miss[b] = 0
                    put(f"[{b}] no domino", frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))
                    continue
            else:
                dom_miss[b] = 0

            # on dessine la boîte détectée
            bx,by,bw,bh = dom_box
            cv2.rectangle(frame, (x1+bx, y1+y0+by), (x1+bx+bw, y1+y0+by+bh), (0,255,128), 2)

            # on stabilise la boîte avec un iou simple
            def iou(a, b_):
                if a is None or b_ is None: 
                    return 0.0
                ax,ay,aw,ah = a; bx_,by_,bw_,bh_ = b_
                xA = max(ax, bx_); yA = max(ay, by_)
                xB = min(ax+aw, bx_+bw_); yB = min(ay+ah, by_+bh_)
                inter = max(0, xB-xA) * max(0, yB-yA)
                union = aw*ah + bw_*bh_ - inter + 1e-9
                return inter/union

            newly_locked = False
            if iou(dom_box_saved[b], dom_box) > 0.5:
                lock_cnt[b] += 1
            else:
                dom_box_saved[b] = dom_box
                lock_cnt[b] = 1
            if lock_cnt[b] >= LOCK_N:
                if not locked[b]:
                    newly_locked = True
                locked[b] = True

            if lock_cnt[b] >= LOCK_N:
                if not locked[b]:
                    newly_locked = True
                locked[b] = True

            # on fige l’image un court instant pour laisser l’OCR lire tranquillement
            if newly_locked:
                global_freeze_until = max(global_freeze_until, time.time() + FREEZE_ON_LOCK_S)
                # on autorise une lecture OCR immédiate pendant le freeze
                next_ocr_t[b] = time.time()


                # === Gating anti-OCR ===
                tile_gray_full = band[by:by+bh, bx:bx+bw]

                # a) flou -> on saute l'OCR
                if cv2.Laplacian(tile_gray_full, cv2.CV_64F).var() < BLUR_VAR_T:
                    # on reporte un peu l'OCR pour cette tuile
                    next_ocr_t[b] = time.time() + 0.20
                    put(f"[{b}] flou", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                    continue

                # b) mouvement -> on attend qu'il se calme
                small = cv2.resize(tile_gray_full, (160, 160), interpolation=cv2.INTER_AREA)
                if prev_tile[b] is None:
                    prev_tile[b] = small.copy()
                    still_cnt[b] = 0
                else:
                    m = np.mean(cv2.absdiff(prev_tile[b], small))
                    prev_tile[b] = small.copy()
                    if m < MOTION_DIFF_T:
                        still_cnt[b] += 1
                    else:
                        still_cnt[b] = 0

                if still_cnt[b] < STILL_FRAMES_FOR_OCR:
                    next_ocr_t[b] = time.time() + 0.15
                    put(f"[{b}] bouge", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                    continue

                # c) main/peau -> on saute l'OCR
                if SKIN_SUPPRESS:
                    # on prend le BGR original, pas le gray
                    tile_bgr = frame[y1+y0+by:y1+y0+by+bh, x1+bx:x1+bx+bw]
                    if tile_bgr.size > 0:
                        ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
                        lower = np.array([0, 133, 77], dtype=np.uint8)
                        upper = np.array([255, 173, 127], dtype=np.uint8)
                        mask = cv2.inRange(ycrcb, lower, upper)
                        if (mask > 0).mean() > SKIN_FRAC_MAX:
                            next_ocr_t[b] = time.time() + 0.20
                            put(f"[{b}] main", frame, (x1+10, y1+20+b*22), 0.55,(255,255,255),1,(0,0,0))
                            continue

                # si on arrive ici, la zone est stable, nette et sans main
                tile_gray = tile_gray_full


            # on lance l’ocr quand on vient de verrouiller ou quand l’intervalle est passé
            cur_val = None
            now = time.time()
            if locked[b] and (newly_locked or now >= next_ocr_t[b]):
                tile_gray = band[by:by+bh, bx:bx+bw]
                cur_val, digits_box = read_number_from_crop(tile_gray)
                next_ocr_t[b] = now + OCR_INTERVAL_S
                if digits_box is not None:
                    dx,dy,dw,dh = digits_box
                    cv2.rectangle(frame, (x1+bx+dx, y1+y0+by+dy),
                                         (x1+bx+dx+dw, y1+y0+by+dy+dh), (0,255,0), 2)

            # on valide la lecture après deux mêmes valeurs
            if cur_val is not None:
                if last_read == cur_val:
                    stable_cnt += 1
                else:
                    last_read  = cur_val
                    stable_cnt = 1
                if stable_cnt >= STABILIZE_N:
                    vals[b] = cur_val
                    target_idx = min(2, target_idx+1)
                    last_read  = None
                    stable_cnt = 0

            put(f"[{b}] {('attend' if vals[b] is None else vals[b])}",
                frame, (x1+10, y1+20+b*22), 0.55, (255,255,255), 1, (0,0,0))

        # on met le hud
        put(f"vals={vals}", frame, (12, 70), 0.9,(255,255,255),2,(0,0,0))
        now = time.time(); fps = 1.0/max(1e-6, now-tprev); tprev = now
        put(f"fps~{int(fps)}", frame, (12, H-12), 0.8,(255,255,255),2,(0,0,0))

        # on affiche et on gère les touches
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        # limite simple des FPS d’affichage
        frame_dt = 1.0 / max(1, TARGET_FPS)
        now2 = time.time()
        elapsed = now2 - now
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

        if k == 27:
            break
        if k == ord('r'):
            # on redéfinit la roi colonne sur l’image actuelle
            rois = pick_two_rois(frame, ROI_FILE)
            col1_norm = rois["col1"]
            vals=[None,None,None]; target_idx=0; last_read=None; stable_cnt=0

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
