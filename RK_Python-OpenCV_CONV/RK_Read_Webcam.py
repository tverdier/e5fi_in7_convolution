"""
Edited by Rostom Kachouri
M1-IRV_ST2IAI _ Mars 2021
"""

"""
Ce programme montre une façon simple de lire et afficher ce que capture la webcam
"""

# Import de la bibliothèque OpenCV
import cv2

# Chargement de la vidéo de la webcam dans la variable cap
# L'argument dans le constructeur de VideoCapture ici est un entier :
# - 0 caméra back (principale)
# - 1 caméra front
# - 2 webcam externe si elle existe 
cameraCapture = cv2.VideoCapture(0)

# Nomme la prochaine fenêtre ouverte 'MyWindow'
cv2.namedWindow('MyWindow')

# Indique dans la console un message qui previent qu'il faut presser une touche pour arrêter la vidéo
print('Showing camera feed. Press any key to stop.')

# Charge une frame de la vidéo dans img et le succès du chargement dans success
success, frame = cameraCapture.read()

# Tant que la frame lue n'est pas vide et que l'utilisateur n'a pas pressé de touche
while success and cv2.waitKey(1) == -1:
    # Affiche la frame chargée précédemment
    cv2.imshow('MyWindow', frame)
    # Charge une frame de la vidéo dans img et le succès du chargement dans success
    success, frame = cameraCapture.read()

# Ferme la fenêtre intitulée 'MyWindow'
cv2.destroyWindow('MyWindow')

# Ferme le fichier vidéo ou la webcam associé à l'objet VideoCapture
cameraCapture.release()

