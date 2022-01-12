"""
Edited by Rostom Kachouri
M1-IRV_ST2IAI _ Mars 2021
"""

"""
Ce programme montre une façon simple de lire et afficher une image
"""

# Import de la bibliothèque OpenCV
import cv2

# Chargement de l'image dans la variable img
img = cv2.imread("RK_Data/lena.jpg")

# Affiche l'image chargée
cv2.imshow("Lena Soderberg",img)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme la fenêtre intitulée 'Lena Soderberg'
cv2.destroyWindow('Lena Soderberg')
