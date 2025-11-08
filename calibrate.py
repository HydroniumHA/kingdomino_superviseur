import cv2
import numpy as np
import json

MATRIX_FILE = 'perspective_matrix.json'
IMAGE_REF_PATH = 'data/reference_image.png' # Adaptez si votre nom de fichier est différent
DEST_SIZE = 600 # Taille de la grille cible (5x5 devient 600x600 pixels)

points_src = [] 
img_original = None
img_display = None

def click_event(event, x, y, flags, param):
    """Callback pour capturer les 4 coins de la zone 5x5."""
    global points_src, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points_src) < 4:
        points_src.append((x, y))
        cv2.circle(img_display, (x, y), 8, (0, 255, 0), -1)
        cv2.imshow("Cliquez sur les 4 coins", img_display)
        print(f"Point {len(points_src)} capturé : ({x}, {y})")

        if len(points_src) == 4:
            print("\nQuatre points capturés. Calcul de la matrice...")
            calculate_and_save_matrix()
            
def calculate_and_save_matrix():
    """Calcule la matrice de perspective (Homographie) et la sauvegarde."""
    
    # 1. Coordonnées sources (clics)
    pts1 = np.float32(points_src)
    
    # 2. Coordonnées de destination (carré parfait 600x600)
    # Ordre : Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche
    pts2 = np.float32([
        [0, 0], 
        [DEST_SIZE - 1, 0], 
        [DEST_SIZE - 1, DEST_SIZE - 1], 
        [0, DEST_SIZE - 1]
    ])

    # 3. Calcul de la Matrice de Transformation (M)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 4. Sauvegarde
    M_list = M.tolist() 
    with open(MATRIX_FILE, 'w') as f:
        json.dump(M_list, f)
        
    print(f"\nMatrice de perspective sauvegardée dans {MATRIX_FILE}")
    
    # 5. Affichage de vérification
    img_warped = cv2.warpPerspective(img_original, M, (DEST_SIZE, DEST_SIZE))
    cv2.imshow('RESULTAT CALIBRATION (Fermez cette fenêtre et la precedente)', img_warped)

# --- Programme Principal ---
if __name__ == "__main__":
    img_original = cv2.imread(IMAGE_REF_PATH) 

    if img_original is None:
        print(f"ERREUR : Impossible de charger l'image de référence. Vérifiez le chemin : {IMAGE_REF_PATH}")
    else:
        # Redimensionnement optionnel pour faciliter les clics
        scale_percent = 50 # 50% de la taille originale
        width = int(img_original.shape[1] * scale_percent / 100)
        height = int(img_original.shape[0] * scale_percent / 100)
        
        img_original = cv2.resize(img_original, (width, height))
        img_display = img_original.copy()
        
        cv2.imshow("Cliquez sur les 4 coins (HG, HD, BD, BG)", img_display)
        cv2.setMouseCallback("Cliquez sur les 4 coins (HG, HD, BD, BG)", click_event)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()