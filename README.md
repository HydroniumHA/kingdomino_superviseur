# Jeu Kingdomino 

Projet de vision par ordinateur pour superviser une partie de **Kingdomino** sur le **plateau 1 (zone centrale)**.

## Objectif – Phase A + Phase B
## PHASE A – Lire et trier les chiffres (dominos)
- On définit 2 zones (ROI) : une pour **COL1** et une pour **COL2** (sélection manuelle, stockée en JSON).
- On coupe chaque colonne en 3 bandes : **haut / milieu / bas**.
- Dans chaque bande, on détecte le domino avec **OpenCV** : amélioration du contraste (**CLAHE**), morphologie (**top-hat / close**), binarisation, puis **contours** pour récupérer une **bounding box** du domino.
- On localise la zone des chiffres dans le domino (contours + filtres aire/forme + fusion si 2 chiffres séparés).
- On lit le nombre avec **EasyOCR** (on upscale, on teste normal/inversé, on garde le meilleur résultat).
- On stabilise la lecture : on valide un chiffre seulement s’il apparaît identique sur plusieurs frames.
- On vérifie le tri : si **haut ≤ milieu ≤ bas** → ordre correct, sinon ordre incorrect.

### **Outils principaux :** OpenCV (détection/bbox), EasyOCR (lecture), NumPy (calculs), time (temporisations/stabilité).
- `reset_state()` : initialise la “mémoire” de lecture (valeurs lues, quelle bande on lit maintenant, stabilité, verrouillage du domino, timers OCR, etc.).
- `is_sorted_ascending(vals)` : vérifie si les 3 nombres sont bien en ordre croissant (haut ≤ milieu ≤ bas).
- `_rect_from_norm_with_pad(...)` : transforme la ROI sauvegardée (en % de l’image) en pixels et l’agrandit (padding) pour mieux couvrir la zone.
- `read_column_sequential(...)` : le cœur :
  - découpe la colonne en 3 bandes (haut/milieu/bas)
  - ne lit qu’une bande à la fois (`target_idx`) pour éviter de se tromper
  - détecte le domino dans la bande (`detect_domino_bbox`) + garde la dernière bbox si ça “clignote” (`KEEP_AFTER_MISS`)
  - “verrouille” le domino si la bbox est stable (`iou`, `lock_cnt`, `LOCK_N`)
  - n’autorise l’OCR que si c’est net (variance Laplacien), immobile (`still_cnt`, `MOTION_DIFF_T`) et sans main (`SKIN_SUPPRESS`)
  - lance l’OCR (`read_number_from_crop`) puis exige une lecture stable (`STABILIZE_N`) avant de valider et passer à la bande suivante.
- `col2_has_any_digits(...)` : en Phase A3, sert à répondre : “est-ce qu’on voit encore des chiffres dans COL2 ?”
  - si la réponse reste “non” assez longtemps → dominos retournés → on passe à la Phase B.

---

## PHASE B – Vérifier l’ordre de jeu des pions (haut -> milieu -> bas)

- On ne fait plus d’OCR : on détecte le mouvement.
- Dans chaque colonne (ROI), on compare deux frames : `diff = abs(current - previous)`.
- On recoupe la ROI en 3 bandes (haut/milieu/bas) et on calcule un score de mouvement par bande (moyenne du diff).
- Une machine à états impose l’ordre :
  - `waiting_pickup` : le pion attendu doit bouger en COL1
  - `carrying` : il doit bouger en COL2
  - `waiting_drop` : on vérifie un temps de calme, puis on valide et on passe au suivant
- Si une autre bande bouge quand ce n’est pas son tour → ordre incorrect (message rouge).

**Outils principaux :** OpenCV (absdiff + blur), NumPy (scores), time (calm_time), logique Python (`TurnManager`).


---

## Fichier principal
- `center_live_PhaseAetB.py`

---

## Dépendances
- Python 3
- OpenCV
- NumPy
- EasyOCR

Installation :
```bash
pip install opencv-python numpy easyocr
