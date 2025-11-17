#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kingdomino – Vérif ordre des pions avec DROP DANS LA 2e COLONNE
#
# Ordre attendu: HAUT -> MILIEU -> BAS
#
# Logique par pion:
#   1. On attend que le pion attendu bouge dans COLONNE 1 (sa bande)
#      => état "carrying"
#   2. Tant qu'on n'a pas vu ce pion arriver dans la même bande de COLONNE 2,
#      on reste sur ce joueur (pas de changement de tour)
#   3. Quand on détecte du mouvement dans la même bande en COLONNE 2,
#      et que ça se stabilise, on dit qu'il a été posé -> on passe au joueur suivant
#
# Ça évite le faux "ERREUR" quand on est encore en train de déplacer le même pion

import json
import os
import time

import cv2
import numpy as np


########################
# CONFIG PAR DÉFAUT
########################

# Ici on fixe la vidéo de test qu'on veut analyser
VIDEO_SOURCE = "/Users/birjan/Documents/MASI/2025-2026/8.Visio/projetJeu/kingdomino/data/center/videos/pions_ok.mp4"

# Quand on filme en live avec le GSM (flux IP)
# IP_SOURCE = "http://192.168.2.147:8080/video"


# On réduit un peu la taille de la vidéo pour afficher plus facilement
SCALE = 0.75

# On peut analyser 1 frame sur N (ici on prend toutes les frames)
STEP = 1

# On garde les coordonnées des deux ROI dans ce fichier JSON
ROI_FILENAME = "roi_config_center.json"

# Paramètres de sensibilité pour la détection de mouvement
MOTION_THR = 6.0    # plus petit = plus sensible
CALM_TIME = 0.8     # temps de calme avant de valider le dépôt
PERSIST_MS = 1700   # temps d’affichage du bandeau OK/ERREUR


########################
# AFFICHAGE
########################

def put(txt, img, org, scale=0.7, color=(255, 255, 255), thick=2, bg=(0, 0, 0)):
    """
    On dessine du texte sur l'image, avec éventuellement un fond rectangulaire
    pour que ce soit lisible
    """
    x, y = org
    if bg is not None:
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        # On dessine un rectangle derrière le texte
        cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + 6), bg, -1)
    # On dessine le texte par-dessus
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_banner(img, text, ok=True, persist_alpha=0.88):
    """
    On dessine un bandeau en haut de l'écran pour afficher OK / ERREUR
    """
    H, W = img.shape[:2]
    # Vert si ok, bleu/rouge si erreur
    color = (60, 180, 75) if ok else (40, 40, 200)
    overlay = img.copy()
    # On dessine un rectangle plein sur la zone du bandeau
    cv2.rectangle(overlay, (0, 0), (W, 56), color, -1)
    # On mélange le bandeau avec l'image originale (effet transparent)
    img[:] = cv2.addWeighted(overlay, persist_alpha, img, 1 - persist_alpha, 0)
    # On écrit le texte par-dessus
    put(text, img, (12, 36), scale=1.15, color=(255, 255, 255), thick=2, bg=None)


########################
# ROI HELPERS
########################

def rect_to_norm(x, y, w, h, W, H):
    """
    On convertit un rectangle en coordonnées normalisées [0..1]
    pour pouvoir réutiliser la ROI même si on change la taille de l'image
    """
    return [x / W, y / H, w / W, h / H]


def norm_to_rect(nx, ny, nw, nh, W, H):
    """
    L'inverse de rect_to_norm : on repasse des coords normalisées aux pixels
    """
    x = int(nx * W)
    y = int(ny * H)
    w = int(nw * W)
    h = int(nh * H)
    return x, y, w, h


def pick_two_rois(first_frame, save_path):
    """
    On laisse l'utilisateur dessiner les 2 ROI à la souris :
    - d'abord la COLONNE 1
    - puis la COLONNE 2
    On sauve ces ROI normalisées dans un fichier JSON pour les réutiliser
    """
    disp = first_frame.copy()
    put("Sélectionne la COLONNE 1 (HAUT->BAS)", disp, (20, 35))
    # On sélectionne la première ROI (colonne 1)
    r1 = cv2.selectROI("Sélection ROI", disp, False, False)  # (x,y,w,h)
    cv2.destroyWindow("Sélection ROI")

    disp = first_frame.copy()
    put("Sélectionne la COLONNE 2 (HAUT->BAS)", disp, (20, 35))
    # On sélectionne la deuxième ROI (colonne 2)
    r2 = cv2.selectROI("Sélection ROI", disp, False, False)
    cv2.destroyWindow("Sélection ROI")

    H, W = first_frame.shape[:2]
    roi = {
        "col1": rect_to_norm(*r1, W, H),
        "col2": rect_to_norm(*r2, W, H),
    }

    # On enregistre ces ROI dans un fichier JSON
    with open(save_path, "w") as f:
        json.dump(roi, f, indent=2)

    print(f"[ROI] Enregistrées dans {save_path} -> {roi}")
    return roi


def load_rois(path):
    """
    On charge les ROI depuis le fichier JSON
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier ROI introuvable: {path}"
        )
    with open(path, "r") as f:
        rois = json.load(f)

    # On vérifie que le fichier contient bien col1 et col2
    if "col1" not in rois or "col2" not in rois:
        raise KeyError("roi_config.json mal formé: clés 'col1' et 'col2' requises.")
    return rois


########################
# DETECTION DE MOUVEMENT PAR BANDE
########################

class RowMotion:
    """
    On détecte le mouvement pour UNE colonne (une ROI)
    On coupe la ROI en 3 bandes horizontales : HAUT, MILIEU, BAS
    À chaque frame :
      - on compare avec la frame précédente
      - on calcule un score de mouvement moyen par bande
      - on retourne:
        * l'indice de la bande la plus active (0,1,2)
        * la valeur max
        * le tableau complet des scores
    """

    def __init__(self, roi_rect, rows=3, thr=6.0):
        # On mémorise la position de la ROI (en pixels)
        self.roi_rect = roi_rect  # (x,y,w,h)
        self.rows = rows
        # seuil qu'on utilisera dans la logique de tour
        self.thr = float(thr)
        # On garde la frame précédente de la ROI pour comparer
        self.prev = None

    def reset_visual_memory(self):
        """
        On réinitialise la mémoire visuelle (utile quand on reset la manche)
        """
        self.prev = None

    def step(self, frame):
        """
        On traite une nouvelle frame :
        - on recadre sur la ROI
        - on passe en niveaux de gris
        - on calcule la différence absolue avec la frame précédente
        - on retourne les scores de mouvement par bande
        """
        x, y, w, h = self.roi_rect
        crop = frame[y:y + h, x:x + w]

        # On travaille en niveaux de gris pour simplifier
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Première frame : on ne peut pas encore calculer de différence
        if self.prev is None:
            self.prev = gray
            return None, None, None

        # On calcule la différence entre cette frame et la précédente
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray

        # On découpe verticalement en "rows" bandes
        rh = h // self.rows
        scores = []
        for i in range(self.rows):
            sub = diff[i * rh:(i + 1) * rh, :]
            # On prend la moyenne de la différence comme score de mouvement
            scores.append(float(sub.mean()))

        scores = np.array(scores)
        idx_max = int(np.argmax(scores))
        val_max = float(scores[idx_max])

        return idx_max, val_max, scores


########################
# LOGIQUE DE TOUR AVEC "PICKUP" + "DROP"
########################

class TurnManager:
    """
    On gère la logique de tour des pions.
    On veut vérifier qu'on suit bien la séquence HAUT -> MILIEU -> BAS.

    Pour chaque pion (ligne) attendu:
      - waiting_pickup : on attend un gros mouvement dans col1[row]
                         -> le joueur prend son pion (pickup)
      - carrying      : on attend ensuite du mouvement dans col2[row]
                         -> le pion arrive dans la seconde colonne
      - waiting_drop  : on attend que la colonne 2 se calme
                         -> pion posé -> on passe au joueur suivant
    """

    def __init__(self, move_threshold=6.0, calm_time=0.8, banner_ms=1700):
        # On commence par le pion du HAUT (ligne 0)
        self.expected_row = 0
        self.sequence_finie = False

        # États possibles pour le pion courant
        self.state = "waiting_pickup"

        # Paramètres pour savoir quand il y a assez de mouvement
        self.move_threshold = float(move_threshold)
        self.calm_time = float(calm_time)
        # On mémorise la dernière fois où ça a beaucoup bougé en colonne 2
        self.last_big_move_time_col2 = 0.0

        # Gestion du bandeau d'information (OK / ERREUR)
        self.last_banner_text = None
        self.last_banner_ok = True
        self.banner_until = 0.0
        # On convertit les millisecondes en secondes
        self.banner_ms = banner_ms / 1000.0

    def row_name(self, r):
        """
        On renvoie le nom de la bande pour l'affichage
        """
        return "HAUT" if r == 0 else ("MILIEU" if r == 1 else "BAS")

    def hud_expected(self):
        """
        On construit le texte pour dire quel pion on attend
        """
        if self.sequence_finie:
            return "Séquence terminée: HAUT -> MILIEU -> BAS"
        return f"Tour attendu: {self.row_name(self.expected_row)}"

    def _next_player(self):
        """
        On passe au pion suivant (HAUT -> MILIEU -> BAS),
        et quand on a fini les 3, on marque la séquence comme terminée
        """
        if self.expected_row < 2:
            self.expected_row += 1
            self.state = "waiting_pickup"
        else:
            self.sequence_finie = True
            self.state = "done"

    def _say_banner(self, txt, ok, now):
        """
        On prépare un nouveau message dans le bandeau
        """
        self.last_banner_text = txt
        self.last_banner_ok = ok
        self.banner_until = now + self.banner_ms

    def get_banner_to_draw(self, now):
        """
        On renvoie le texte du bandeau s'il doit encore être affiché
        """
        if now < self.banner_until and self.last_banner_text:
            return self.last_banner_text, self.last_banner_ok
        return None, None

    def reset_all(self):
        """
        On remet tout à zéro pour recommencer une nouvelle manche
        """
        self.expected_row = 0
        self.sequence_finie = False
        self.state = "waiting_pickup"
        self.last_banner_text = None
        self.banner_until = 0.0
        self.last_big_move_time_col2 = 0.0

    def update(self, now, idx1, val1, scores1, idx2, val2, scores2):
        """
        On met à jour l'état du tour en fonction des mouvements détectés
        - scores1 : mouvement par bande dans la colonne 1
        - scores2 : mouvement par bande dans la colonne 2
        """
        if self.sequence_finie or self.state == "done":
            return

        cur_row = self.expected_row
        cur_name = self.row_name(cur_row)

        # 1) waiting_pickup : on attend le mouvement du joueur attendu en col1
        if self.state == "waiting_pickup":
            if scores1 is not None:
                # On regarde si la bande du joueur attendu bouge assez
                if scores1[cur_row] is not None and scores1[cur_row] >= self.move_threshold:
                    # Le pion a été pris -> on passe en mode "carrying"
                    self.state = "carrying"
                    self._say_banner(f"{cur_name} commence son tour -> OK (pickup)", True, now)
                    return

                # On détecte si un joueur FUTUR joue trop tôt
                for r in [0, 1, 2]:
                    if r == cur_row:
                        continue
                    if r < cur_row:
                        continue  # ce joueur a déjà joué
                    if scores1[r] is not None and scores1[r] >= self.move_threshold:
                        # On signale une erreur : mauvais ordre
                        self._say_banner(
                            f"{self.row_name(r)} bouge alors que c'est {cur_name} -> ERREUR",
                            False,
                            now,
                        )
                        return
            # Si pas de mouvement intéressant, on ne change rien
            return

        # 2) carrying : on attend l'apparition du pion en col2[cur_row]
        if self.state == "carrying":
            if scores2 is not None:
                # On attend du mouvement dans la même bande mais en colonne 2
                if scores2[cur_row] is not None and scores2[cur_row] >= self.move_threshold:
                    # Le pion arrive dans la deuxième colonne
                    self.state = "waiting_drop"
                    self.last_big_move_time_col2 = now
                    self._say_banner(
                        f"{cur_name} posé dans colonne 2 ? (checking stabilité)",
                        True,
                        now,
                    )
                    return
            return

        # 3) waiting_drop : on attend que la colonne 2 se stabilise
        if self.state == "waiting_drop":
            if scores2 is not None:
                # On met à jour le temps du dernier gros mouvement
                cur_score2 = scores2[cur_row] if scores2[cur_row] is not None else 0.0
                if cur_score2 >= self.move_threshold:
                    self.last_big_move_time_col2 = now

            # On regarde depuis combien de temps c'est calme
            time_since_last_big = now - self.last_big_move_time_col2
            if time_since_last_big >= self.calm_time:
                # On valide le pion et on passe au suivant
                self._say_banner(f"{cur_name} VALIDÉ -> Prochain joueur", True, now)
                self._next_player()
            return


########################
# MAIN LOOP
########################

def main():
    # On choisit la source (vidéo ou caméra)
    src = VIDEO_SOURCE
    
    # On choisit la source : ici on prend le flux live du GSM
    # src = IP_SOURCE

    # On récupère les paramètres
    scale = SCALE
    step = STEP

    # On construit le chemin complet vers le fichier ROI
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roi_file = os.path.join(script_dir, ROI_FILENAME)

    motion_thr = MOTION_THR
    calm_time = CALM_TIME
    persist_ms = PERSIST_MS

    # --- Ouverture vidéo/cam
    # Si on met "0", on ouvre la webcam, sinon on ouvre le fichier vidéo
    cap = cv2.VideoCapture(0 if isinstance(src, str) and src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d’ouvrir {src}")

    # On lit la première frame pour connaître la taille et choisir les ROI
    ok, frame0 = cap.read()
    if not ok:
        print("Frame initiale vide")
        return

    H0, W0 = frame0.shape[:2]
    # On applique le scale
    W = int(W0 * scale)
    H = int(H0 * scale)
    frame0s = cv2.resize(frame0, (W, H))

    # --- ROI
    # Si le fichier ROI n'existe pas encore, on demande à l'utilisateur
    if not os.path.exists(roi_file):
        print(f"[ROI] Fichier inexistant -> sélection manuelle des colonnes.")
        rois = pick_two_rois(frame0s, roi_file)
    else:
        # Sinon on recharge simplement les ROI
        rois = load_rois(roi_file)

    col1_norm = rois["col1"]
    col2_norm = rois["col2"]

    # On convertit les ROI normalisées en pixels sur la taille actuelle
    x1, y1, w1, h1 = norm_to_rect(*col1_norm, W, H)
    x2, y2, w2, h2 = norm_to_rect(*col2_norm, W, H)

    # --- Détecteurs de mouvement pour chaque colonne
    motion_col1 = RowMotion((x1, y1, w1, h1), rows=3, thr=motion_thr)
    motion_col2 = RowMotion((x2, y2, w2, h2), rows=3, thr=motion_thr)

    # --- Manager des tours (logique HAUT -> MILIEU -> BAS)
    tm = TurnManager(
        move_threshold=motion_thr,
        calm_time=calm_time,
        banner_ms=persist_ms,
    )

    # --- Fenêtre d'affichage OpenCV
    win = "Kingdomino – Ordre des pions (avec colonne 2)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 980, int(980 * H / W))

    t_prev = time.time()
    i = 0

    # Boucle principale de lecture des frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # On peut sauter des frames pour aller plus vite si step > 1
        if step > 1 and (i % step) != 0:
            i += 1
            continue
        i += 1

        # On redimensionne la frame pour correspondre au SCALE
        frame = cv2.resize(frame, (W, H))

        # On recalcule les ROI en pixels (sécurité si la taille change)
        x1, y1, w1, h1 = norm_to_rect(*col1_norm, W, H)
        x2, y2, w2, h2 = norm_to_rect(*col2_norm, W, H)
        motion_col1.roi_rect = (x1, y1, w1, h1)
        motion_col2.roi_rect = (x2, y2, w2, h2)

        # On calcule le mouvement dans la colonne 1 et la colonne 2
        idx1, val1, scores1 = motion_col1.step(frame)
        idx2, val2, scores2 = motion_col2.step(frame)

        now = time.time()
        # On met à jour la logique de tour
        tm.update(now, idx1, val1, scores1, idx2, val2, scores2)

        # On dessine les deux ROI sur l'image
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)

        # On calcule les FPS pour info
        fps = 1.0 / max(1e-6, now - t_prev)
        t_prev = now
        put(f"fps~{int(fps)}", frame, (12, H - 12), 0.8, (255, 255, 255), 2, (0, 0, 0))

        # On affiche quel pion on attend (HAUT / MILIEU / BAS)
        put(tm.hud_expected(), frame, (12, 70), 0.75, (255, 255, 255), 2, (0, 0, 0))

        # On affiche éventuellement le dernier bandeau OK / ERREUR
        banner_text, banner_ok = tm.get_banner_to_draw(now)
        if banner_text:
            draw_banner(frame, banner_text, ok=banner_ok)

        # On montre le résultat dans la fenêtre
        cv2.imshow(win, frame)

        # Gestion du clavier
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC pour quitter
            break
        if k == ord('r'):
            # On remet tout à zéro pour rejouer une nouvelle manche
            tm.reset_all()
            motion_col1.reset_visual_memory()
            motion_col2.reset_visual_memory()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # On lance la boucle principale si on exécute ce fichier directement
    main()
