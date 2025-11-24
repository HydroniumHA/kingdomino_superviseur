#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Jeu Kingdomino
# on a 2 grandes phases A et B

# PHASE A  mise en place du round
#   on vérifie que la colonne 1 est triée croissant
#   on vérifie que la colonne 2 est triée croissant
#   on attend que la colonne 2 ne montre plus de chiffres donc les dominos sont retournés
#   même si ce n est pas trié on continue quand même

# PHASE B  ordre des joueurs et déplacement des pions
#   on attend HAUT puis MILIEU puis BAS
#   on valide un joueur quand son pion est bien posé et stable dans la colonne 2
#   si un joueur futur bouge trop tôt on affiche un bandeau rouge

import cv2, time, json, argparse, os
import numpy as np
import easyocr


############################################################
# === UTIL AFFICHAGE TEXTE & BANDEAUX ======================
############################################################

# fonction pratique pour écrire du texte lisible sur l image
# ajout d un fond pr que le texte se voit bien

def put(txt, img, org, scale=0.7, color=(255,255,255), thick=2, bg=(0,0,0)):
    """
    petit helper pour écrire du texte lisible  option fond noir
    """
    x, y = org
    if bg is not None:
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (x-4, y-h-6), (x+w+4, y+6), bg, -1)
    cv2.putText(
        img, txt, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA
    )


# on dessine un bandeau en haut pr donner un retour visuel rapide
# vert si tout va bien rouge si erreur

def draw_banner(img, text, ok=True, persist_alpha=0.88):
    """
    bandeau en haut  vert si ok rouge si erreur
    """
    H, W = img.shape[:2]
    color = (60,180,75) if ok else (40,40,200)  # BGR
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (W, 56), color, -1)
    img[:] = cv2.addWeighted(overlay, persist_alpha, img, 1-persist_alpha, 0)
    put(text, img, (12, 36),
        scale=1.15, color=(255,255,255), thick=2, bg=None)


############################################################
# === ROI HELPERS  sauvegarde en coordonnées normalisées ===
############################################################

# on convertit les rectangles en coordonnées normalisées pour les sauvegarder
# comme ça on peut redimensionner la vidéo et garder les mêmes zones

def rect_to_norm(x, y, w, h, W, H):
    """
    je convertis des coordonnées pixel -> normalisées 0..1
    pour pouvoir sauvegarder dans un JSON et réutiliser
    même si l image est redimensionnée
    """
    return [x / W, y / H, w / W, h / H]


# on refait l opération inverse normalisé vers pixels pour recadrer

def norm_to_rect(nx, ny, nw, nh, W, H):
    """
    je convertis coord normalisées -> pixels
    pour dessiner ou recadrer dans la frame redimensionnée
    """
    x = int(nx * W)
    y = int(ny * H)
    w = int(nw * W)
    h = int(nh * H)
    return x, y, w, h


# on laisse l utilisateur dessiner les deux zones une fois et on les sauvegarde

def pick_two_rois(first_frame, save_path):
    """
    on dessine ROI COLONNE 1 puis ROI COLONNE 2
    COLONNE 1 = les 3 dominos ou pions de gauche
    COLONNE 2 = les 3 dominos ou pions de droite
    """
    disp = first_frame.copy()
    put("Selectionne la COLONNE 1 HAUT->BAS", disp, (20, 35))
    r1 = cv2.selectROI("Selection ROI", disp, False, False)  # x y w h
    cv2.destroyWindow("Selection ROI")

    disp = first_frame.copy()
    put("Selectionne la COLONNE 2 HAUT->BAS", disp, (20, 35))
    r2 = cv2.selectROI("Selection ROI", disp, False, False)
    cv2.destroyWindow("Selection ROI")

    H, W = first_frame.shape[:2]
    roi = {
        "col1": rect_to_norm(*r1, W, H),
        "col2": rect_to_norm(*r2, W, H)
    }

    with open(save_path, "w") as f:
        json.dump(roi, f, indent=2)

    print(f"[ROI] Enregistrees dans {save_path} -> {roi}")
    return roi


# on recharge les ROI depuis le JSON si elles existent déjà

def load_rois(path):
    """
    lis le fichier roi_config.json
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier ROI introuvable: {path} utilise --pick_roi"
        )
    with open(path, "r") as f:
        rois = json.load(f)

    if "col1" not in rois or "col2" not in rois:
        raise KeyError("roi_config.json mal forme  cles col1 et col2 requises")

    return rois


############################################################
# === OCR nombres avec EasyOCR  PHASE A uniquement =========
############################################################

# on initialise le lecteur EasyOCR une seule fois pour gagner du temps
_reader = None

def get_reader():
    """
    init EasyOCR une seule fois
    """
    global _reader
    if _reader is None:
        print("Init EasyOCR CPU…")
        _reader = easyocr.Reader(['en'], gpu=False)  # CPU ça suffit
    return _reader


# on lit les nombres présents dans une ROI en niveaux de gris
# on agrandit avant l OCR pour mieux voir les petits chiffres

def read_numbers(gray_roi, min_conf=0.6):
    """
    lis les chiffres verticaux détectés dans gray_roi
    on upscale x2 pour aider EasyOCR
    retourne nums et leurs boîtes triées de haut en bas
    """
    reader = get_reader()

    # on agrandit pour mieux lire les petits numéros
    big = cv2.resize(
        gray_roi, None,
        fx=2.0, fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    res = reader.readtext(
        big,
        detail=1,
        paragraph=False,
        text_threshold=min_conf
    )

    nums, boxes = [], []
    for (bbox, text, conf) in res:
        # on garde seulement les chiffres
        t = ''.join([c for c in text if c.isdigit()])
        if t.strip() == '':
            continue
        try:
            v = int(t)
            nums.append(v)
            boxes.append(bbox)
        except:
            pass

    # on trie de haut en bas pour avoir l ordre visuel
    if boxes:
        ys = [np.mean([p[1] for p in b]) for b in boxes]
        order = np.argsort(ys)
        nums = [nums[i] for i in order]
        boxes = [boxes[i] for i in order]

    return nums, boxes


# on vérifie si on a exactement 3 nombres et s ils sont triés strictement croissant

def is_sorted_strict(nums):
    """
    True si les 3 chiffres sont triés croissant strict
    None si pas exactement 3 nombres
    """
    if len(nums) != 3:
        return None
    return nums == sorted(nums)


############################################################
# === DÉTECTION DE MOUVEMENT PAR BANDE POUR PHASE B ========
############################################################

# on détecte le mouvement dans chaque colonne en la coupant en 3 bandes
# on prend la moyenne des différences par bande pour savoir où ça bouge le plus

class RowMotion:
    """
    Détecteur de mouvement pour UNE colonne  col1 ou col2

    on découpe la ROI en 3 sous bandes verticales empilées
        0 = HAUT
        1 = MILIEU
        2 = BAS

    à chaque frame on calcule la différence avec la frame précédente
    et on retourne la bande la plus active et les scores
    """
    def __init__(self, roi_rect, rows=3, thr=6.0):
        self.roi_rect = roi_rect   # x y w h
        self.rows = rows
        self.thr = float(thr)
        self.prev = None

    def reset_visual_memory(self):
        # on efface la frame précédente pour repartir propre
        self.prev = None

    def step(self, frame):
        # on recadre sur la colonne
        x, y, w, h = self.roi_rect
        crop = frame[y:y+h, x:x+w]

        # on passe en gris et on lisse un peu
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # on initialise la mémoire au premier passage
        if self.prev is None:
            self.prev = gray
            return None, None, None

        # on calcule la différence absolue
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray

        # on coupe en 3 bandes et on mesure l activité
        rh = h // self.rows
        scores = []
        for i in range(self.rows):
            sub = diff[i*rh:(i+1)*rh, :]
            scores.append(float(sub.mean()))

        scores = np.array(scores)
        idx_max = int(np.argmax(scores))
        val_max = float(scores[idx_max])

        return idx_max, val_max, scores


############################################################
# === GESTION DU TOUR DES JOUEURS  PHASE B =================
############################################################

# on gère la petite machine à états pour le joueur attendu
# waiting_pickup -> carrying -> waiting_drop -> next player

class TurnManager:
    """
    Gestion du tour d un pion avec 3 étapes
      waiting_pickup -> carrying -> waiting_drop -> next player

    on affiche un bandeau quand on valide ou quand on détecte une erreur
    """
    def __init__(self,
                 move_threshold=6.0,
                 calm_time=0.8,
                 banner_ms=1700):
        self.expected_row = 0        # on commence par HAUT
        self.sequence_finie = False

        # états possibles pour le pion courant
        # waiting_pickup  carrying  waiting_drop  done
        self.state = "waiting_pickup"

        self.move_threshold = float(move_threshold)
        self.calm_time = float(calm_time)

        # on mémorise le dernier gros mouvement en colonne 2
        self.last_big_move_time_col2 = 0.0

        # bandeau interne au TurnManager
        self.last_banner_text = None
        self.last_banner_ok = True
        self.banner_until = 0.0
        self.banner_ms = banner_ms / 1000.0

    def row_name(self, r):
        return "HAUT" if r == 0 else ("MILIEU" if r == 1 else "BAS")

    def hud_expected(self):
        if self.sequence_finie:
            return "Sequence terminee  HAUT -> MILIEU -> BAS"
        return f"Tour attendu  {self.row_name(self.expected_row)}"

    def _next_player(self):
        # on passe au joueur suivant ou on finit la séquence
        if self.expected_row < 2:
            self.expected_row += 1
            self.state = "waiting_pickup"
        else:
            self.sequence_finie = True
            self.state = "done"

    def _say_banner(self, txt, ok, now):
        # on prépare un bandeau à afficher pendant un court temps
        self.last_banner_text = txt
        self.last_banner_ok = ok
        self.banner_until = now + self.banner_ms

    def get_banner_to_draw(self, now):
        """
        donne texte et couleur de bandeau à dessiner si actif
        sinon None None
        """
        if now < self.banner_until and self.last_banner_text:
            return self.last_banner_text, self.last_banner_ok
        return None, None

    def reset_all(self):
        """
        reset manuel touche r pour relancer juste la PHASE B
        """
        self.expected_row = 0
        self.sequence_finie = False
        self.state = "waiting_pickup"
        self.last_banner_text = None
        self.banner_until = 0.0
        self.last_big_move_time_col2 = 0.0

    def update(self, now,
               idx1, val1, scores1,
               idx2, val2, scores2):
        """
        idx1 et scores1 mouvement colonne 1  joueurs qui prennent leur pion
        idx2 et scores2 mouvement colonne 2  zone où on pose le pion
        scores sont des tableaux de 3 valeurs  haut milieu bas
        """
        if self.sequence_finie or self.state == "done":
            return

        cur_row = self.expected_row
        cur_name = self.row_name(cur_row)

        # waiting_pickup
        # on veut voir bouger la bande du joueur attendu en colonne 1
        # si un joueur futur bouge trop tôt on affiche erreur
        if self.state == "waiting_pickup":
            if scores1 is not None:
                # le joueur attendu commence à bouger
                if scores1[cur_row] >= self.move_threshold:
                    self.state = "carrying"
                    self._say_banner(
                        f"{cur_name} commence son tour -> OK pickup",
                        True, now
                    )
                    return

                # sinon on vérifie si un joueur futur bouge trop tôt
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

        # carrying
        # le joueur transporte le pion et on attend une activité en colonne 2
        if self.state == "carrying":
            if scores2 is not None:
                if scores2[cur_row] >= self.move_threshold:
                    self.state = "waiting_drop"
                    self.last_big_move_time_col2 = now
                    self._say_banner(
                        f"{cur_name} pose dans colonne 2  on verifie la stabilite",
                        True, now
                    )
                    return
            return

        # waiting_drop
        # le pion doit rester calme un petit moment en colonne 2
        if self.state == "waiting_drop":
            if scores2 is not None:
                if scores2[cur_row] >= self.move_threshold:
                    self.last_big_move_time_col2 = now

            time_since_last_big = now - self.last_big_move_time_col2
            if time_since_last_big >= self.calm_time:
                self._say_banner(
                    f"{cur_name} VALIDE -> prochain joueur",
                    True, now
                )
                self._next_player()
            return


############################################################
# === MAIN ================================================
############################################################

# on gère les arguments en ligne de commande
# on ouvre la vidéo ou la caméra
# on charge ou on dessine les ROI
# on exécute PHASE A puis PHASE B dans une boucle temps réel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="http://192.168.2.147:8080/video",
                    help='chemin video mp4 mov flux reseau ex http://192.168.2.147:8080/video ou index camera 0')
    ap.add_argument("--scale", type=float, default=0.75,
                    help="resize 0.75 = plus petit donc plus rapide")
    ap.add_argument("--step", type=int, default=1,
                    help="analyser 1 frame sur N  1 = tout analyser")
    ap.add_argument("--roi_file", default="src/roi_config.json",
                    help="JSON avec col1 et col2")
    ap.add_argument("--pick_roi", action="store_true",
                    help="redemander les 2 ROI au lancement")
    # OCR phase A
    ap.add_argument("--min_conf", type=float, default=0.55,
                    help="confiance mini OCR")
    ap.add_argument("--stable_needed", type=int, default=8,
                    help="# frames OK d affilee avant de valider l ordre")
    ap.add_argument("--persist_ms", type=int, default=1700,
                    help="duree du bandeau ms")

    # mouvement phase B
    ap.add_argument("--motion_thr", type=float, default=6.0,
                    help="seuil mouvement plus petit = plus sensible")
    ap.add_argument("--calm_time", type=float, default=0.8,
                    help="temps calme en col2 avant de valider le depot")

    args = ap.parse_args()

    # OUVERTURE VIDEO ou CAM
    # on accepte une URL un fichier vidéo ou un index de caméra
    src = args.source
    cap = cv2.VideoCapture(0 if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d ouvrir {src}")

    ok, frame0 = cap.read()
    if not ok:
        print("frame initiale vide")
        return

    # on calcule les dimensions de travail en fonction du scale
    H0, W0 = frame0.shape[:2]
    W = int(W0 * args.scale)
    H = int(H0 * args.scale)
    frame0s = cv2.resize(frame0, (W, H))

    # ROI CHARGEMENT ou SELECTION
    # on choisit les zones une fois puis on les réutilise
    if args.pick_roi:
        rois = pick_two_rois(frame0s, args.roi_file)
    else:
        rois = load_rois(args.roi_file)

    col1_norm = rois["col1"]   # colonne gauche
    col2_norm = rois["col2"]   # colonne droite

    # Préparation des états pour PHASE A et PHASE B

    # PHASE A
    phase_global = "verif_col1"  # puis verif_col2 puis attente_flip puis pions
    stable_cnt = 0               # nombre de frames où l ordre est bon
    last_banner_text_A = None
    last_banner_ok_A = True
    banner_until_A = 0.0

    # PHASE B
    # on crée deux détecteurs de mouvement pour chaque colonne
    x1, y1, w1, h1 = norm_to_rect(*col1_norm, W, H)
    x2, y2, w2, h2 = norm_to_rect(*col2_norm, W, H)

    motion_col1 = RowMotion((x1,y1,w1,h1), rows=3, thr=args.motion_thr)
    motion_col2 = RowMotion((x2,y2,w2,h2), rows=3, thr=args.motion_thr)

    tm = TurnManager(
        move_threshold=args.motion_thr,
        calm_time=args.calm_time,
        banner_ms=args.persist_ms
    )

    # fenêtre d affichage
    win = "Kingdomino – Setup + Ordre des pions"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 980, int(980 * H / W))

    t_prev = time.time()
    i = 0

    # on boucle sur les frames
    while True:
        ok, frame_raw = cap.read()
        if not ok:
            break

        # on saute des frames si demandé pour aller plus vite
        if args.step > 1 and (i % args.step) != 0:
            i += 1
            continue
        i += 1

        # on met la frame à la bonne taille
        frame = cv2.resize(frame_raw, (W, H))

        # on recalcule les ROI en pixels au cas où
        x1, y1, w1, h1 = norm_to_rect(*col1_norm, W, H)
        x2, y2, w2, h2 = norm_to_rect(*col2_norm, W, H)

        # on met à jour les rectangles internes des détecteurs
        motion_col1.roi_rect = (x1,y1,w1,h1)
        motion_col2.roi_rect = (x2,y2,w2,h2)

        # texte d aide en bas de l écran
        hud_text = ""

        ########################################################
        # PHASE A  OCR + setup du tour
        ########################################################
        if phase_global in ("verif_col1", "verif_col2"):
            # on choisit la colonne à vérifier
            if phase_global == "verif_col1":
                cx, cy, cw, ch = (x1, y1, w1, h1)
                col_id = 1
            else:
                cx, cy, cw, ch = (x2, y2, w2, h2)
                col_id = 2

            crop = frame[cy:cy+ch, cx:cx+cw]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # on lit les nombres détectés et on teste si triés
            nums, _ = read_numbers(gray, min_conf=args.min_conf)

            if len(nums) == 3:
                ok_sorted = is_sorted_strict(nums)
                if ok_sorted:
                    stable_cnt += 1  # on cumule les frames stables
                else:
                    stable_cnt = 0

                hud_text = (
                    f"Verif COL {col_id} | nums={nums} | "
                    f"ordre={'OK' if ok_sorted else 'KO'}"
                )

                # on valide après un certain nombre de frames stables
                if stable_cnt >= args.stable_needed:
                    last_banner_text_A = (
                        f"COL {col_id} – ORDRE "
                        f"{'CORRECT' if ok_sorted else 'INCORRECT'}"
                    )
                    last_banner_ok_A = bool(ok_sorted)
                    banner_until_A = time.time() + (args.persist_ms / 1000.0)
                    stable_cnt = 0

                    # on avance de phase même si ce n est pas trié
                    if phase_global == "verif_col1":
                        phase_global = "verif_col2"
                    else:
                        phase_global = "attente_flip"

            else:
                # pas encore 3 nombres clairs
                stable_cnt = 0
                hud_text = (
                    f"Verif COL {col_id} | nums={nums} | ordre=NA pas 3 nbres lisibles"
                )

        elif phase_global == "attente_flip":
            # on attend que la colonne 2 ne montre plus de chiffres
            # cela veut dire que les dominos sont retournés
            crop2 = frame[y2:y2+h2, x2:x2+w2]
            g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
            nums2, _ = read_numbers(g2, min_conf=args.min_conf)

            hud_text = f"Attente FLIP col2 | nums={nums2}"

            if len(nums2) == 0:
                # prêt pour jouer la phase B
                last_banner_text_A = "COL 2 – RETOURNEE  OK on peut jouer"
                last_banner_ok_A = True
                banner_until_A = time.time() + 1.0

                phase_global = "pions"

                # on réinitialise les mémoires des détecteurs
                motion_col1.reset_visual_memory()
                motion_col2.reset_visual_memory()
                tm.reset_all()  # on commence HAUT puis MILIEU puis BAS

        ########################################################
        # PHASE B  ordre des joueurs
        ########################################################
        elif phase_global == "pions":
            # on calcule le mouvement dans les deux colonnes
            idx1, val1, scores1 = motion_col1.step(frame)
            idx2, val2, scores2 = motion_col2.step(frame)

            now = time.time()
            tm.update(
                now,
                idx1, val1, scores1,
                idx2, val2, scores2
            )

            hud_text = tm.hud_expected()

        else:
            # état inconnu sécurité
            hud_text = f"phase={phase_global}"

        ########################################################
        # DESSIN  HUD  BANDEAUX
        ########################################################

        # on dessine les 2 colonnes sur l image
        cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)  # gauche
        cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (255,0,0), 2)  # droite

        # on calcule un fps simple pour savoir la fluidité
        now = time.time()
        fps = 1.0 / max(1e-6, now - t_prev)
        t_prev = now

        put(f"phase={phase_global} | fps~{int(fps)}",
            frame,
            (12, H-12),
            0.8, (255,255,255), 2, (0,0,0))

        if hud_text:
            put(hud_text,
                frame,
                (12, 70),
                0.75, (255,255,255), 2, (0,0,0))

        # on affiche le bandeau de phase A si actif
        if phase_global != "pions":
            if (time.time() < banner_until_A) and last_banner_text_A:
                draw_banner(frame, last_banner_text_A, ok=last_banner_ok_A)
        else:
            # sinon on prend le bandeau éventuel de la phase B
            btxt, bok = tm.get_banner_to_draw(time.time())
            if btxt:
                draw_banner(frame, btxt, ok=bok)

        # on montre la fenêtre vidéo
        cv2.imshow(win, frame)

        # on gère les touches clavier
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('r'):
            # on reset juste la phase B si on est dedans
            if phase_global == "pions":
                tm.reset_all()
                motion_col1.reset_visual_memory()
                motion_col2.reset_visual_memory()
                tm.last_banner_text = None
                tm.banner_until = 0.0

    # fin boucle
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
