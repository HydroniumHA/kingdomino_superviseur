import cv2
import numpy as np
import time

framewidth = 640
frameheight = 480
# cap = cv2.VideoCapture("./data/output_fast.mp4")
cap = cv2.VideoCapture("http://192.168.178.153:8080/video")
# cap.set(3, framewidth)
# cap.set(4, frameheight)
print("FPS : " + str(cap.get(cv2.CAP_PROP_FPS)))

TARGET_FPS = 5.0
FRAME_INTERVAL = 1.0 / TARGET_FPS


def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 0, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 0, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 100000, empty)

cv2.setTrackbarPos("Threshold1", "Parameters", 100) 
cv2.setTrackbarPos("Threshold2", "Parameters", 40)
cv2.setTrackbarPos("Area", "Parameters", 15000) # Adapter l'aire pour la "taille" du carré du chateau.

def stackImages(scale, imgArray):
    """
    Empile plusieurs images dans une seule fenêtre pour affichage.
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width =imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank] *rows
        for x in range (0,rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) ==2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 2)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) # approx c'est les points du contour
            # print(len(approx))

            # Pour récuperer les cordonnées des points :
            # for point in approx:
            #     x, y = point[0]  # Extractions des coordonnées x et y
            #     print(f"Point: ({x}, {y})")
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # Appeler la nouvelle fonction pour afficher et traiter l'aire
            displayArea(imgContour, cnt, area)

def displayArea(img, cnt, area):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_img = img[y:y+h, x:x+w]

    # Afficher l'image rognée dans une nouvelle fenêtre
    cv2.imshow("Last Detected Tile", cropped_img)

    # Appeler la fonction processTile avec l'image rognée et l'aire
    # processTile(cropped_img, area)

def processBiome(cropped_img, area):
    print("Processing biome with area:", area)
    # afficher une modal de choix de biome (herbe, eau, foret, champ, mine) dans le terminal
    print("Choisissez le biome pour cette tuile :")
    print("1. Herbe")
    print("2. Eau")
    print("3. Forêt")
    print("4. Champ")
    print("5. Mine")
    choix = input("Entrez le numéro du biome choisi : ")
    biome_dict = {
        "1": "Herbe",
        "2": "Eau",
        "3": "Forêt",
        "4": "Champ",
        "5": "Mine"
    }
    biome = biome_dict.get(choix, "Inconnu")
    print(f"Biome choisi : {biome} pour une aire de {area}")

# Variable globale pour stocker les informations du royaume
royaume = {
    "area": 0,
    "coordinates": None,
    "last_detected_time": None  # Temps de la dernière détection
}

# Variable globale pour les informations du château
chateau = {
    "area": 0,
    "coordinates": None,
    "last_detected_time": None  # Temps de la dernière détection
}

nouvelle_tuiele = {
    "area": 0,
    "coordinates": None,
    "last_detected_time": None  # Temps de la dernière détection
}

CONFIRMATION_TIME = 3  # Temps de confirmation en secondes
 

def detectChateau(img, contours):
    global chateau
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            x, y, w, h = cv2.boundingRect(cnt)
            if chateau["coordinates"] is None:
                # Démarrer le chronomètre pour confirmer le château
                if chateau["last_detected_time"] is None:
                    chateau["last_detected_time"] = time.time()
                    print("Château détecté, en attente de confirmation...")
                elif time.time() - chateau["last_detected_time"] >= CONFIRMATION_TIME:
                    # Confirmer le château après le délai
                    chateau["area"] = area
                    chateau["coordinates"] = (x, y, w, h)
                    chateau["last_detected_time"] = None
                    print(f"Château confirmé : Aire = {area}, Coordonnées = {x, y, w, h}")
            return cnt
    return None

def isolateTile(img, contours):
    global royaume
    global chateau

    if royaume["coordinates"] is None:
        print("Aucun royaume détecté. Impossible d'isoler les tuiles.")
        return

    royaume_x, royaume_y, royaume_w, royaume_h = royaume["coordinates"]
    royaume_area = royaume["area"]
    chateau_x, chateau_y, chateau_w, chateau_h = chateau["coordinates"]
    area_chateau = chateau["area"]

    print(f"Royaume actuel : Aire = {royaume_area}, Coordonnées = {royaume['coordinates']}")

    # cv2.imshow("Royaume", img[royaume_y:royaume_y+royaume_h, royaume_x:royaume_x+royaume_w])

    # cordonnée du royaume : royaume_x, royaume_y, royaume_w, royaume_h
    TOLERANCE = 50  # Tolérance en pixels pour les coordonnées    
    # cordonnée du royaume - 100 pixels pour avoir une marge
    # royaume_x = int(royaume_x + TOLERANCE / 2)
    # royaume_y = int(royaume_y + TOLERANCE / 2)
    # royaume_w = int(royaume_w - TOLERANCE)
    # royaume_h = int(royaume_h - TOLERANCE)

    # affichage du royaume avec marge et sans marges
    # imgStack = stackImages(0.5, ([img[royaume_y:royaume_y+royaume_h, royaume_x:royaume_x+royaume_w]], [img[royaume_y-TOLERANCE//2:royaume_y+royaume_h+TOLERANCE//2, royaume_x-TOLERANCE//2:royaume_x+royaume_w+TOLERANCE//2]]))
    # cv2.imshow("Royaume avec/sans marge", imgStack)
    # cv2.imshow("Royaume avec marge", img[royaume_y:royaume_y+royaume_h, royaume_x:royaume_x+royaume_w])
    
    # on affiche l'image que displayAndProcessArea(imgContour, cnt, area) affiche mais en soustraiant la partie du royaume qui existe déjà

    TILE_CONFIRMATION_TIME = 3  # Temps de confirmation en secondes

    for cnt in contours:
        area = cv2.contourArea(cnt)
        nouvelle_tuile_area = area - royaume_area
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

        # Vérifier si l'aire de la nouvelle tuile est au moins 1,75 fois plus grande que celle du château
        if nouvelle_tuile_area > area_chateau * 1.75:
            print("Nouvelle tuile potentielle détectée avec aire :", nouvelle_tuile_area, " (Château aire :", area_chateau, ")")

            ## On sait dedecter la pause d'une nouvelle tuile. Mainteant objectif : isoler la tuile (donc area - royaume_area) et ses coordonnées. 
            # + commencer timer

            # encadrer l'area de la nouvelle_tuile_area (donc les cordonnées de area - les cordonnnées de royaume) dans un nouveau cv2.imshow appelée nouvelle_tuile.
            
            # 1. Créer un masque noir de la taille de l'image
            mask_total = np.zeros(img.shape[:2], dtype=np.uint8)
            mask_royaume = np.zeros(img.shape[:2], dtype=np.uint8)

            # 2. Dessiner le contour détecté actuellement (Royaume + Tuile) en blanc sur le premier masque
            cv2.drawContours(mask_total, [cnt], -1, 255, -1)

            # 3. Dessiner l'ancien royaume (rectangle connu) en blanc sur le deuxième masque
            cv2.rectangle(mask_royaume, 
                          (royaume_x, royaume_y), 
                          (royaume_x + royaume_w, royaume_y + royaume_h), 
                          255, -1)

            # 4. Soustraction : Masque Total - Masque Royaume = Masque de la nouvelle tuile
            mask_tuile_only = cv2.subtract(mask_total, mask_royaume)

            # 5. Trouver les contours de cette différence pour obtenir les coordonnées de la tuile seule
            contours_tuile, _ = cv2.findContours(mask_tuile_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_tuile:
                # On prend le plus grand contour trouvé (au cas où il y aurait du bruit)
                largest_cnt_tuile = max(contours_tuile, key=cv2.contourArea)
                
                # Récupérer les coordonnées (x, y, w, h) spécifiques à la nouvelle tuile
                nouvelle_tuile_x, nouvelle_tuile_y, nouvelle_tuile_w, nouvelle_tuile_h = cv2.boundingRect(largest_cnt_tuile)


                # Vérifier que les dimensions sont valides (>0)
                if nouvelle_tuile_w > 0 and nouvelle_tuile_h > 0:
                    # Découper l'image originale avec ces nouvelles coordonnées
                    crop_tuile = img[nouvelle_tuile_y:nouvelle_tuile_y+nouvelle_tuile_h, nouvelle_tuile_x:nouvelle_tuile_x+nouvelle_tuile_w]
                    if nouvelle_tuiele["last_detected_time"] is None:
                        nouvelle_tuiele["last_detected_time"] = time.time()
                        print("Nouvelle tuile détectée, en attente de confirmation...")
                    elif time.time() - nouvelle_tuiele["last_detected_time"] >= TILE_CONFIRMATION_TIME:
                        # Afficher
                        cv2.imshow("nouvelle_tuile", crop_tuile)
                        
                        # Optionnel : Dessiner un rectangle vert autour de la nouvelle tuile sur l'image principale
                        cv2.rectangle(img, (nouvelle_tuile_x, nouvelle_tuile_y), (nouvelle_tuile_x + nouvelle_tuile_w, nouvelle_tuile_y + nouvelle_tuile_h), (0, 255, 0), 2)

                        # Stocker les informations de la nouvelle tuile
                        nouvelle_tuiele["area"] = nouvelle_tuile_area
                        nouvelle_tuiele["coordinates"] = (nouvelle_tuile_x, nouvelle_tuile_y, nouvelle_tuile_w, nouvelle_tuile_h)

                        print("Nouvelle tuile détectée et confirmée avec aire :", nouvelle_tuile_area, "Coordonnées :", nouvelle_tuiele["coordinates"])

                        # Appeler la fonction de traitement de la tuile (elle met à jour le royaume et reinialise la tuile après traitement)
                        processTile(crop_tuile, nouvelle_tuile_area)



                


def processTile(cropped_img, area):
    # Mettre à jour le royaume avec les nouvelles informations et réinitialiser la nouvelle_tuile["last_detected_time"] = None afin de permettre la dedection d'une nouvelle tuiel
    print("Processing tile with area:", area)
    global royaume
    global nouvelle_tuiele
    processBiome(cropped_img, area)
    # Mettre à jour le royaume avec les informations de la nouvelle tuile
    royaume["area"] = nouvelle_tuiele["area"]
    royaume["coordinates"] = nouvelle_tuiele["coordinates"]
    print("Royaume mis à jour avec la nouvelle tuile : Aire =", royaume["area"], "Coordonnées =", royaume["coordinates"])
    # Réinitialiser la nouvelle_tuile
    nouvelle_tuiele["area"] = 0
    nouvelle_tuiele["coordinates"] = None
    nouvelle_tuiele["last_detected_time"] = None


def main():
    global royaume
    '''
    1. Dedecter les contours du chateau. Il y aura une tollerence d'aires et de cordonnées.
        Si il n'y a pas de changement dans les 5 secondes mettre l'aire ainsi que les coordonnées du chateau détecté dans une variable globale nommée royaume.
    2. Chaque tours :
        a. Si changement de dedection > tollerence commencer le compteur de 5 secondes.
            Apres les 5 secondes, isoler la tuile fraichement posée (avec son aire) grace aux contours dedectés et aux coordonnées du royaume.
            On soustrait l'aire du chateau à l'aire totale dedectée pour obtenir l'aire de la tuile posée ainsi que ses coordonnées.
            Afficher la tuile 
        b. Appeler la fonction processTile(cropped_img, area) pour process la tuile posée et on met à jour le royaume avec les nouvelles informations (aire et cordonnées).
    
    '''
    game_timer_start = time.time()

    last_process_time = time.time()
    while True:
        success, img = cap.read()
        if not success:
            print("Erreur : Impossible de lire la vidéo")
            break

        # Limite la fréquence de traitement (à TARGET_FPS)
        current_time = time.time()
        if (current_time - last_process_time) >= FRAME_INTERVAL:
            start_process_time = time.time()
            
            imgContour = img.copy()
            imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            
            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
            getContours(imgDil, imgContour) # TODO : Essayer sans cette fonction, elle affiche last dedected tile window mais je crois qu'elle fait rien d'autre
            contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


            ### Logique 


            # Étape 1 : Détecter les contours du château avec une tolérance d'aires et de coordonnées.
            # Si aucun changement n'est détecté dans les 3 secondes, enregistrer l'aire et les coordonnées du château dans la variable globale 'royaume'.
            if chateau["coordinates"] is None:
                print("CONDITION IF CHATEAU[\"coordinates\"] NONE TRUE")
                # Appel de la fonction pour détecter le château au début, celle ci est appelee a chaque frame jusqu'a ce que le chateau soit detecté
                detectChateau(imgContour, contours)
                royaume = chateau 
                
            # Étape 2 : Chaque tour, vérifier les changements de détection.
            else:
                # print("CONDITION SI CHATEAU & ROYAUME EST INITIALISE ")
                isolateTile(imgContour, contours)

                # b. Appeler la fonction processTile pour traiter la tuile posée et mettre à jour le royaume avec les nouvelles informations.
                # (Cette logique est incluse dans la fonction isolateTile.)

            imgStack = stackImages(0.5, ([imgContour]))
            cv2.imshow("Result", imgStack)
            
            # MAJ le temps du dernier processing
            last_process_time = current_time
            # Print la durée du processing

            # end_process_time = time.time()
            # processing_duration = (end_process_time - start_process_time) * 1000  # Pour convertir en ms
            # print(f"Processing Duration: {processing_duration:.3f} milliseconds")

        # Sortir de la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()