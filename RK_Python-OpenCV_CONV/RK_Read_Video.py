"""
Edited by Rostom Kachouri
M1-IRV_ST2IAI _ Mars 2021
"""

"""
Ce programme montre une façon simple de lire et afficher une vidéo
"""

# Import de la bibliothèque OpenCV
import cv2

# Définition des dimensions "par défaut" d'une frame
#frameWidth = 640
#frameHeight = 480

# Chargement de la vidéo nommé 'test_ video.mp4' située dans le dossier 'RK_Data' dans la variable cap
cap = cv2.VideoCapture("RK_Data/test_ video.mp4")

# On entre dans une boucle infinie
while True:
    # Charge une frame de la vidéo dans img et le succès du chargement dans success
    success, img = cap.read()

    # Redimensionne l'image avec les dimensions frameWidth x frameHeight
    #img = cv2.resize(img, (frameWidth, frameHeight))

    # Affiche la frame
    cv2.imshow("Video", img)

    # Attend une milliseconde
    #cv2.waitKey(1)
    
    # Attend une milliseconde et vérifie que la touche appuyée par l'utilisateur
    if cv2.waitKey(1) == ord('q'):
        # Si l'utilisateur a pressé la touche Q, on sort de la boucle infinie
        break

# Ferme la fenêtre intitulée 'Video'
cv2. destroyWindow ('Video')



