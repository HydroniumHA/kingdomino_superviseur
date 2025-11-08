import cv2
import numpy as np
import json

# --- Configuration Générale ---
VIDEO_PATH = 'data/partie_joueur.mp4'
MATRIX_FILE = 'perspective_matrix.json'
DEST_SIZE = 600 # Taille corrigée du plateau (600x600)
TUIL_SIZE = DEST_SIZE // 5 # Taille de chaque case 1x1 (120x120)

# --- Paramètres de Détection de Présence (à ajuster !) ---
PRESENCE_THRESHOLD = 25 # Seuil d'écart-type (variance) pour séparer la table des tuiles
FRAMES_REQUIRED_FOR_VALIDATION = 300 # Frames consécutives pour confirmer qu'une tuile est posée (anti-main)

# --- Modèle Virtuel du Royaume ---
VIRTUAL_SIZE = 9
# VIRTUAL_KINGDOM stocke l'état permanent (0=Vide, 1=Occupé)
VIRTUAL_KINGDOM = np.zeros((VIRTUAL_SIZE, VIRTUAL_SIZE), dtype=int) 
# CONFIRMATION_COUNT suit l'historique des détections pour la validation temporelle
CONFIRMATION_COUNT = np.zeros((VIRTUAL_SIZE, VIRTUAL_SIZE), dtype=int) 

# Variables d'état
PERSPECTIVE_MATRIX = None
cap = None # La variable de capture sera globale pour le logging

def load_perspective_matrix():
    """Charge la matrice de transformation pour la correction."""
    global PERSPECTIVE_MATRIX
    try:
        with open(MATRIX_FILE, 'r') as f:
            M_list = json.load(f)
            PERSPECTIVE_MATRIX = np.float32(M_list)
        print(f"✅ Matrice de perspective chargée.")
        return True
    except FileNotFoundError:
        print(f"❌ ERREUR : Fichier {MATRIX_FILE} introuvable. Exécutez d'abord calibrate.py.")
        return False

def is_tile_present(roi):
    """
    Détermine si une case (ROI) est occupée en utilisant la variance de luminosité.
    """
    # 1. Conversion en niveaux de gris
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculer l'écart type (variance) de la luminosité
    std_dev = np.std(gray_roi)
    
    return std_dev > PRESENCE_THRESHOLD

def process_frame(frame):
    """
    Applique la correction, détecte la présence, gère le royaume virtuel et valide la pose.
    """
    global VIRTUAL_KINGDOM, CONFIRMATION_COUNT, cap
    
    if PERSPECTIVE_MATRIX is None:
        return frame
        
    warped_frame = cv2.warpPerspective(frame, PERSPECTIVE_MATRIX, (DEST_SIZE, DEST_SIZE))
    
    OFFSET = (VIRTUAL_SIZE - 5) // 2 # Décalage pour centrer la zone 5x5 dans le 9x9 (OFFSET=2)
    newly_placed_tiles = []
    
    # --- 1. Détection et Validation Temporelle ---
    for i in range(5): # Colonnes pixel 0-4
        for j in range(5): # Lignes pixel 0-4
            
            # Coordonnées pixel de la ROI (dans le 600x600)
            x_start, y_start = i * TUIL_SIZE, j * TUIL_SIZE
            x_end, y_end = (i + 1) * TUIL_SIZE, (j + 1) * TUIL_SIZE
            roi = warped_frame[y_start:y_end, x_start:x_end]
            
            # Coordonnées dans la matrice 9x9
            virtual_x, virtual_y = i + OFFSET, j + OFFSET
            
            
            # A. Détection Brute (Main ou Tuile)
            if is_tile_present(roi):
                # INC : Si présence détectée, incrémenter le compteur, seulement si la case est VIDE
                if VIRTUAL_KINGDOM[virtual_y, virtual_x] == 0: 
                    CONFIRMATION_COUNT[virtual_y, virtual_x] += 1
            else:
                # Si rien n'est présent (main partie), réinitialiser le compteur
                CONFIRMATION_COUNT[virtual_y, virtual_x] = 0

            # B. Validation de la Pose (Validation Temporelle)
            if CONFIRMATION_COUNT[virtual_y, virtual_x] >= FRAMES_REQUIRED_FOR_VALIDATION and VIRTUAL_KINGDOM[virtual_y, virtual_x] == 0:
                
                # C'est ici que la tuile est jugée permanente !
                
                # 1. Marquer la case comme OCCUPÉE de manière PERMANENTE
                VIRTUAL_KINGDOM[virtual_y, virtual_x] = 1 
                newly_placed_tiles.append((virtual_x, virtual_y))
                
                # 2. Remise à zéro
                CONFIRMATION_COUNT[virtual_y, virtual_x] = 0

                # 3. Mise en évidence (Rouge - Nouvelle Pose Validée)
                cv2.rectangle(warped_frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 4)
                cv2.putText(warped_frame, "VALIDEE!", (x_start + 5, y_start + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            # C. Visualisation
            # Tuiles déjà posées (Vert)
            elif VIRTUAL_KINGDOM[virtual_y, virtual_x] == 1:
                cv2.rectangle(warped_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
            # Affichage du statut de confirmation (pour le débug de la validation)
            if CONFIRMATION_COUNT[virtual_y, virtual_x] > 0:
                 cv2.putText(warped_frame, f"Conf: {CONFIRMATION_COUNT[virtual_y, virtual_x]}", (x_start + 5, y_start + 70), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

            # Dessin de la grille
            cv2.rectangle(warped_frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)
    
    # --- 4. Journalisation de l'événement ---
    if newly_placed_tiles:
        # A ce stade, la tuile est considérée comme VRAIMENT posée.
        print(f"*** NOUVELLE POSE VALIDÉE (temps : {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.2f}s) : {newly_placed_tiles} ***")
        # ICI : Insérer la vérification des règles de connexion et 5x5 !
        
    return warped_frame


def main():
    global cap
    if not load_perspective_matrix():
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Erreur : Impossible d'ouvrir la vidéo à l'emplacement {VIDEO_PATH}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break
            
        processed_frame = process_frame(frame)
        
        cv2.imshow('Kingdomino Supervisieur (Grille 5x5 Corrigee)', processed_frame)
        
        # Le temps d'attente affecte la vitesse de lecture de la vidéo et la validation temporelle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()