import cv2
import numpy as np
import time

framewidth = 640
frameheight = 480
# cap = cv2.VideoCapture("./data/output_fast.mp4")
cap = cv2.VideoCapture("http://192.168.2.226:8080/video")
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
            displayAndProcessArea(imgContour, cnt, area)

def displayAndProcessArea(img, cnt, area):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_img = img[y:y+h, x:x+w]

    # Afficher l'image rognée dans une nouvelle fenêtre
    cv2.imshow("Last Detected Tile", cropped_img)

    # Appeler la fonction processTile avec l'image rognée et l'aire
    # processTile(cropped_img, area)

def processBiome(cropped_img, area):
    print("Processing biome with area:", area)

# Variable globale pour stocker les informations du royaume
royaume = {
    "area": 0,
    "coordinates": None,
    "last_detected_time": None  # Temps de la dernière détection
}

CONFIRMATION_TIME = 3  # Temps de confirmation en secondes

def detectChateau(img, contours):
    global royaume
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            x, y, w, h = cv2.boundingRect(cnt)
            if royaume["coordinates"] is None:
                # Démarrer le chronomètre pour confirmer le château
                if royaume["last_detected_time"] is None:
                    royaume["last_detected_time"] = time.time()
                    print("Château détecté, en attente de confirmation...")
                elif time.time() - royaume["last_detected_time"] >= CONFIRMATION_TIME:
                    # Confirmer le château après le délai
                    royaume["area"] = area
                    royaume["coordinates"] = (x, y, w, h)
                    royaume["last_detected_time"] = None
                    print(f"Château confirmé : Aire = {area}, Coordonnées = {x, y, w, h}")
            return cnt  # Retourne le contour du château
    return None

# Ajout d'une variable globale pour la sensibilité
SENSITIVITY_FACTOR = 0.5  # Facteur réglable pour la détection des nouvelles tuiles

def isolateTile(img, contours):
    global royaume
    if royaume["coordinates"] is None:
        print("Aucun royaume détecté. Impossible d'isoler les tuiles.")
        return

    # Tolérance dynamique basée sur la taille du château
    DYNAMIC_TOLERANCE = max(50, int(0.1 * (royaume["coordinates"][2] + royaume["coordinates"][3])))
    TOLERANCE_COORDS = DYNAMIC_TOLERANCE  # Tolérance pour les coordonnées (en pixels)
    TOLERANCE_AREA = 1000   # Tolérance pour l'aire

    royaume_x, royaume_y, royaume_w, royaume_h = royaume["coordinates"]
    royaume_area = royaume["area"]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Vérifier si le contour correspond à une tuile en excluant le royaume avec tolérance
        if area > SENSITIVITY_FACTOR * royaume_area and \
           abs(area - royaume_area) > TOLERANCE_AREA and \
           not (royaume_x - TOLERANCE_COORDS <= x <= royaume_x + royaume_w + TOLERANCE_COORDS and
                royaume_y - TOLERANCE_COORDS <= y <= royaume_y + royaume_h + TOLERANCE_COORDS):
            
            current_time = time.time()
            cropped_img = img[y:y+h, x:x+w]

            if royaume["last_detected_time"] is None:
                # Démarrer le chronomètre pour la nouvelle tuile
                royaume["last_detected_time"] = current_time
                print("Nouvelle tuile détectée, en attente de confirmation...")

                # Afficher l'image fixe de la tuile détectée
                cv2.imshow("Detected Tile (Awaiting Confirmation)", cropped_img)
            elif current_time - royaume["last_detected_time"] >= CONFIRMATION_TIME:
                # Confirmer la tuile après le délai
                print(f"Tuile confirmée après {CONFIRMATION_TIME} secondes.")
                nouvelle_tuile_area = area - royaume_area  # Calculer l'aire de la nouvelle tuile
                print(f"Aire de la nouvelle tuile : {nouvelle_tuile_area}")

                # Mettre à jour les coordonnées et l'aire du royaume
                royaume["area"] += nouvelle_tuile_area
                royaume["coordinates"] = (
                    min(royaume_x, x),
                    min(royaume_y, y),
                    max(royaume_x + royaume_w, x + w) - min(royaume_x, x),
                    max(royaume_y + royaume_h, y + h) - min(royaume_y, y)
                )
                royaume["last_detected_time"] = None  # Réinitialiser le temps de détection
                print(f"Royaume mis à jour : Aire = {royaume['area']}, Coordonnées = {royaume['coordinates']}")

                # Afficher la nouvelle tuile confirmée
                cv2.imshow("Confirmed Tile", cropped_img)
            else:
                print(f"En attente de confirmation... {current_time - royaume['last_detected_time']:.2f} secondes écoulées.")

            # Afficher la tuile détectée
            displayAndProcessArea(img, cnt, area)

def processTile(cropped_img, area):
    print("Processing tile with area:", area)
    # Ajoutez ici le traitement spécifique pour la tuile

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
            getContours(imgDil, imgContour)

            ### Logique 

            contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if royaume["coordinates"] is None:
                # Détecter le château au début
                detectChateau(imgContour, contours)
            else:
                isolateTile(imgContour, contours)

            imgStack = stackImages(0.5, ([imgContour]))
            cv2.imshow("Result", imgStack)
            
            # MAJ le temps du dernier processing
            last_process_time = current_time
            # Print la durée du processing

            # end_process_time = time.time()
            # processing_duration = (end_process_time - start_process_time) * 1000  # Pour convertir en ms
            # print(f"Processing Duration: {processing_duration:.3f} milliseconds")

        if cv2.waitKey(1) & 0xFF == 27:  # Escape key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()