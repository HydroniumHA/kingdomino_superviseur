import cv2
import numpy as np

IMAGE_REF_PATH = 'data/reference_image.png' # Image du plateau avec des tuiles
selected_points = []
img_original = None

def mouse_callback(event, x, y, flags, param):
    """Callback pour capturer les clics et analyser la couleur."""
    global img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Définir une petite zone (par exemple 20x20 pixels) autour du clic pour l'analyse
        # Cela réduit l'influence du bruit et des lignes de la grille.
        ROI_SIZE = 10
        x_start = max(0, x - ROI_SIZE)
        y_start = max(0, y - ROI_SIZE)
        x_end = min(img_original.shape[1], x + ROI_SIZE)
        y_end = min(img_original.shape[0], y + ROI_SIZE)

        roi = img_original[y_start:y_end, x_start:x_end]
        
        # Convertir en HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculer la moyenne
        mean_hsv = cv2.mean(hsv_roi)[:3] # [H, S, V]
        
        # Afficher la valeur
        print(f"\n--- Coordonnées ({x}, {y}) ---")
        print(f"HUE (Teinte): {mean_hsv[0]:.2f} (0-179)")
        print(f"SATURATION: {mean_hsv[1]:.2f} (0-255)")
        print(f"VALUE (Valeur): {mean_hsv[2]:.2f} (0-255)")
        
        # Dessiner un point pour le feedback visuel
        cv2.circle(img_original, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("HSV Analyser (Cliquez sur une tuile)", img_original)
        
if __name__ == "__main__":
    
    # Chargez une image fixe (ou un frame corrigé de la vidéo)
    img_original = cv2.imread(IMAGE_REF_PATH) 

    if img_original is None:
        print(f"ERREUR : Impossible de charger l'image de référence à {IMAGE_REF_PATH}")
    else:
        cv2.imshow("HSV Analyser (Cliquez sur une tuile)", img_original)
        cv2.setMouseCallback("HSV Analyser (Cliquez sur une tuile)", mouse_callback)
        
        print("Cliquez sur différents types de terrain pour obtenir leurs plages HSV.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()