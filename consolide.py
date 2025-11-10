import cv2
import numpy as np
import os

# --- Configuration ---
VIDEO_PATH = 'data/partie_joueur_2.mp4'

# --- Données globales ---
CHATEAU_TEMPLATES = []
castle_reference_frame = None
g_castle_contour = None
g_chateau_w = 0  # Largeur du château en pixels après détection du template
g_chateau_h = 0  # Hauteur du château en pixels après détection du template
VIRTUAL_KINGDOM = np.full((5, 5), "VIDE", dtype='<U8')

# --- Paramètres ---
TEMPLATE_MATCH_THRESHOLD = 0.7
DIFF_THRESHOLD = 15
TUILE_SIZE_ERROR_MARGIN = 0.33  # ±33% de tolérance

# --- Chargement simple des templates château ---
def load_chateau_templates():
    global CHATEAU_TEMPLATES
    paths = ['data/chateau1.png', 'data/chateau2.png', 'data/chateau3.png']
    for path in paths:
        if os.path.exists(path):
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                CHATEAU_TEMPLATES.append(template)
                print(f"✅ Template chargé: {path}")
    print(f"✅ {len(CHATEAU_TEMPLATES)} templates château chargés.")

# --- Détection du château + mesure de sa taille ---
def find_or_track_castle(frame):
    global g_castle_contour, castle_reference_frame, g_chateau_w, g_chateau_h
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_val = -1
    best_loc = None
    best_shape = None

    for template in CHATEAU_TEMPLATES:
        h, w = template.shape
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_shape = (w, h)

    if best_val >= TEMPLATE_MATCH_THRESHOLD and best_loc is not None:
        x, y = best_loc
        w, h = best_shape
        g_chateau_w, g_chateau_h = w, h
        
        g_castle_contour = np.array([[[x, y]], [[x, y+h]], [[x+w, y+h]], [[x+w, y]]], dtype=np.int32)
        
        if castle_reference_frame is None:
            castle_reference_frame = frame.copy()
            VIRTUAL_KINGDOM[2, 2] = "CHATEAU"
            print("✅ Château détecté et référence sauvegardée.")
        
        cv2.polylines(frame, [g_castle_contour], True, (0, 255, 255), 2)
        cv2.putText(frame, "CHATEAU", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return True
    return False

# --- Détecter les nouvelles tuiles sur TOUTE l'image ---
def detect_new_tiles(current_frame):
    if castle_reference_frame is None or g_chateau_w == 0:
        return []
    
    # Comparaison sur toute l'image
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(castle_reference_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_curr, gray_ref)
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = []

    base_w, base_h = g_chateau_w, g_chateau_h
    tol_w = base_w * TUILE_SIZE_ERROR_MARGIN
    tol_h = base_h * TUILE_SIZE_ERROR_MARGIN

    valid_sizes = [
        (base_w, base_h),           # 1x1
        (base_w, base_h * 2),       # 1x2 (vertical)
        (base_w * 2, base_h),       # 2x1 (horizontal)
    ]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 20000:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtrer forme allongée excessive (éviter bruit)
        ratio = max(w, h) / min(w, h)
        if ratio > 2.5:
            continue

        matched = False
        for exp_w, exp_h in valid_sizes:
            if (abs(w - exp_w) <= tol_w * (exp_w / base_w) and
                abs(h - exp_h) <= tol_h * (exp_h / base_h)):
                matched = True
                break
        
        if matched:
            tiles.append((x, y, w, h))  # Coordonnées globales
    
    return tiles

# --- Validation immédiate ---
def update_tile_confirmation(new_tiles):
    return new_tiles

# --- Ajout dans la matrice ---
def add_tile_to_kingdom(bbox):
    for i in range(5):
        for j in range(5):
            if VIRTUAL_KINGDOM[i, j] == "VIDE":
                VIRTUAL_KINGDOM[i, j] = "TUILE"
                return

# --- Affichage matrice ---
def draw_kingdom_on_frame(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (200, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    start_x, start_y = 10, 30
    line_height = 25
    cv2.putText(frame, "ROYAUME:", (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for y in range(5):
        row_text = " ".join(VIRTUAL_KINGDOM[y, x][:4].ljust(4) for x in range(5))
        cv2.putText(frame, row_text, (start_x, start_y + (y+1)*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# --- Fonction principale ---
def main():
    global castle_reference_frame
    load_chateau_templates()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir {VIDEO_PATH}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        if find_or_track_castle(display_frame):
            tiles = detect_new_tiles(frame)
            for bbox in update_tile_confirmation(tiles):
                add_tile_to_kingdom(bbox)
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(display_frame, "TUILE", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        draw_kingdom_on_frame(display_frame)
        cv2.imshow('King Domino - Supervision', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎯 ROYAUME FINAL:")
    print(VIRTUAL_KINGDOM)

if __name__ == "__main__":
    main()