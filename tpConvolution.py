# -*- coding: utf-8 -*-
"""
Edited by Céline NGUYEN, Pierre LEBAS et Thibault VERDIER
TP Convolution
"""

# Import de la bibliothèque OpenCV
import cv2
# Import de la bibliothèque Numpy
import numpy as np
# Import de l'objet Tuple qui servira à représenter un vecteur 3D (couleur)
from typing import Tuple


print("2 - Opérations de base\n")
###################################################################################################

def createRgbColorImg(width=1, height=1, rgb_color=(0, 0, 0)):
    """
    Fonction générant un tableau numpy 3D de <width> pixel(s) par <height> pixel(s) de couleur <rgb_color>
    Appeler la fonction sans argument, revient à générer un pixel noir.
    
    Args:
        width (int, optionnel): longueur / nombre de pixel de gauche à droite. Par défaut à 1.
        height (int, optionnel): largeur / hauteur / nombre de pixel de haut en bas. Par défaut à 1.
        rgb_color (tuple, optionnel): couleur RGB sous forme d'un tuple de 3 dimensions. Par défaut à (0, 0, 0).

    Returns:
    np array: tableau numpy 3D de <width> pixel(s) par <height> pixel(s) de couleur <rgb_color>
    """
    
    # Création d'une image noire
    image = np.zeros((height, width, 3), np.uint8)

    # OpenCV utilise le format BGR (Blue, Green, Red) pour encoder les pixels
    # Conversion d'une couleur RGB en une couleur BGR
    color = tuple(reversed(rgb_color))

    # Applique la couleur BGR à chaque pixel de l'image
    image[:] = color

    return image


print("2.1 - Création et manipulation d’images avec Python et OpenCV\n")

# Crée une image noire de 128x128
img_black = createRgbColorImg(128, 128)

# Affiche l'image noire
cv2.imshow("Image Noire 128x128", img_black)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres ouvertes
cv2.destroyAllWindows()


print("2.2 - Modifiez ce code pour afficher trois images de taille 512*512 respectivement de couleur Rouge, Verte et Bleue\n")

# Définition des 3 images Rouge, Verte et Bleue
img_red = createRgbColorImg(512, 512, (255, 0, 0))
img_green = createRgbColorImg(512, 512, (0, 255, 0))
img_blue = createRgbColorImg(512, 512, (0, 0, 255))

# Affichage des 3 Images
cv2.imshow("Image Rouge 512x512", img_red)
cv2.imshow("Image Verte 512x512", img_green)
cv2.imshow("Image Bleue 512x512", img_blue)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres ouvertes
cv2.destroyAllWindows()


print("2.3 - Insérez des lignes, rectangles, cercles et du texte dans ces images. Pour cela, les fonctions «cv2.line», «cv2.rectangle», «cv2.circle», et «cv2.putText» seront utilisées. Vous devez identifier et fixer les paramètres de ces fonctions\n")

# Définition d'une image noire
img_ligne_rectangle_cercle = createRgbColorImg(512, 512)

# Dessine la diagonale rouge d'en haut à gauche jusqu'en bas à droite
cv2.line(img_ligne_rectangle_cercle, (0, 0), (512, 512), (0, 0, 255))

# Dessine un rectangle au centre de l'image
cv2.rectangle(img_ligne_rectangle_cercle, (128, 128), (384, 384), (0, 255, 0))

# Dessine un cercle au centre de l'image de rayon 128
cv2.circle(img_ligne_rectangle_cercle, (256, 256), 128, (255, 0, 0))

# Dessine un texte au centre de l'image
cv2.putText(img_ligne_rectangle_cercle, 'texte', (220, 256), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 125, 0))

# Affichage de l'image avec la ligne, le cercle et le rectangle
cv2.imshow("Image avec ligne, rectangle, cercle et texte", img_ligne_rectangle_cercle)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres ouvertes
cv2.destroyAllWindows()

###################################################################################################




print("3 - Traitement d'images\n")
###################################################################################################

print("3.2 Conversion Couleur\n")

print("3.2.1 - Pour convertir une image couleur en une image niveau de gris, on applique souvent la formule suivante : # Gray = 0.2989 R + 0.5870 G + 0.1140 B\n")
print("Ecrire un programme permettant de convertir une entrée couleur (Image / Vidéo / Webcam) en niveau de gris et afficher deux fenêtres pour visualiser l’entrée et la sortie.\n")

imgPath = "RK_Python-OpenCV_CONV/RK_Data/"
imgName = "baboon.png"
imgFullPath = imgPath + imgName

# Chargement de l'image dans la variable img
# Le deuxième argument de la fonction imread est un flag qui indique le mode de lecture de l'image
#   - cv2.IMREAD_COLOR permet de charger l'image en respectant les couleurs de celle-ci (format BGR)
#   - cv2.IMREAD_GRAYSCALE permet de charger l'image en transformant les couleurs en niveau de gris
img_color = cv2.imread(imgFullPath, cv2.IMREAD_COLOR)
img_grey = cv2.imread(imgFullPath, cv2.IMREAD_GRAYSCALE)

def cvtInGray(imgSrc):
    """
    Convertit une image couleur en une image en niveau de gris à partir de la formule G = 0.2989*R + 0.5870*G + 0.1140*B

    Args:
        imgSrc (np array): Image en couleur

    Returns:
        np array: Image convertit en niveau de gris à partir de la formule G = 0.2989*R + 0.5870*G + 0.1140*B
    """    
    # Crée une copie de l'image chargée en couleur
    imgDes = imgSrc.copy()

    # Récupération de le nombre de pixel dans la longueur de l'image
    width = len(imgSrc)
    
    # Pour chaque ligne..
    for x in range(width):
        height = len(imgSrc[x])
        # .. et pour chaque colonne
        for y in range(height):
            
            # Récupère la valeur de chaque channel du pixel aux coordonnées [x;y]
            # Au format BGR, le bleu est la première chaîne et le rouge est la dernière
            blue_channel_value = imgSrc[x][y][0]
            green_channel_value = imgSrc[x][y][1]
            red_channel_value = imgSrc[x][y][2]
            
            # Applique la formule souvent utilisé pour convertir une image en niveau de gris
            # Soit G = 0.2989*R + 0.5870*G + 0.1140*B
            imgDes[x][y] = 0.2989*red_channel_value + 0.5870*green_channel_value + 0.1140*blue_channel_value

    return imgDes

img_color_greyscaled = cvtInGray(img_color)

# Affiche les images
cv2.imshow("Image en couleur", img_color)
cv2.imshow("Image en niveau de gris", img_grey)
cv2.imshow("Image convertit en niveau de gris", img_color_greyscaled)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres
cv2.destroyAllWindows()


print("3.2.2 - La fonction OpenCV « cv2.cvtColor » permet d’assurer ce type de conversion vers d’autre espace couleur (HSV par exemple en employant le paramètre « cv2.COLOR_BGR2HSV ») ou bien vers un espace niveau de gris (en employant le paramètre « cv2.COLOR_BGR2GRAY »).\n")
print("Modifiez le programme précédent pour afficher d’autres fenêtres pour illustrer les résultats de conversion des mêmes entrées.\n")
print("Comparez les résultats de votre développement et ceux obtenus par la fonction « cv2.cvtColor ».\n")

# Convertit l'image couleur stockée dans img_color dans l'espace HSV
img_color_cvt_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
img_color_cvt_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Affiche les images dans les différents espaces de couleur
cv2.imshow("Image en niveau de gris", img_grey)
cv2.imshow("Image convertit dans l'espace HSV", img_color_cvt_hsv)
cv2.imshow("Image convertit en niveau de gris avec cvtColor", img_color_cvt_gray)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres
cv2.destroyAllWindows()


print("3.2 Convolution d'images\n")

# Définition d'une fonction qui applique un filtre moyenneur 5x5 à une image en niveau de gris 
def meanFilter(imgSrc, kX=1, kY=1):
    """
    Applique un filtre moyenneur kX*kY à imgSrc et renvoie un np array qui correspond à l'image filtrée

    Args:
        kX (int, optionnel): nombre de pixel en longueur prise en compte dans la moyenne. Par défaut à 1.
        kY (int, optionnel): nombre de pixel en longueur prise en compte dans la moyenne. Par défaut à 1.
        
    Returns:
        np array: tableau numpy 3D qui correspond à l'image filtrée
    """
    
    # Copie de l'image source
    imgDes = imgSrc.copy()
    
    # Récupération de le nombre de pixel dans la longueur de l'image
    width = len(imgSrc)
    # Pour chaque ligne..
    for x in range(width):
        # .. on récupère la taille de la colonne
        height = len(imgSrc[x])
        # .. et pour chaque colonne..
        for y in range(height):
    
            # Initialise la somme des valeurs des pixels utilisés
            pixelValueSum = 0
            # Initialise le nombre de pixel utilisé
            nbPixelRead = 0
            
            # Pour i allant de 0 à kX..
            for i in range(kX):
                # .. et pour j allant de 0 à kX
                for j in range(kY):
                    # Vérifie qu'on ne sort pas de l'image
                    if (0 <= x+i and x+i < width) and (0 <= y+j and y+j < height):
                        # Ajoute à la somme la valeur du pixel [x+i; y+j]
                        pixelValueSum += imgSrc[x+i][y+j]
                        # Incrémente le nombre de pixel(s) utilisé(s)
                        nbPixelRead += 1
    
            # Attribue une nouvelle valeur à l'image filtrée
            # Celle-ci correspond à la la moyenne des valeurs des pixels utilisés
            imgDes[x][y] = pixelValueSum/nbPixelRead
    
    return imgDes

print("Filtre moyenneur\n")

# Applique un filtre moyenneur 5x5 à l'image en niveau de gris
img_color_cvt_gray_meanFiltered = meanFilter(img_color_cvt_gray, 5, 5)
img_color_cvt_gray_blurred = cv2.blur(img_color_cvt_gray, (5, 5))

# Affiche les images
cv2.imshow("Image convertit en niveau de gris", img_color_cvt_gray)
cv2.imshow("Image convertit floutee avec filtre moyenneur 5x5", img_color_cvt_gray_meanFiltered)
cv2.imshow("Image convertit et floutee avec la fonction blur", img_color_cvt_gray_blurred)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres
cv2.destroyAllWindows()



# Définition d'un fonction qui applique le filtre de Sobel à une image en niveau de gris
def sobelFilter(imgSrc):
    """
    Applique le filtre Sobel à une image en niveau de gris

    Args:
        imgSrc (np array): tableau numpy représentant une image en niveau de gris à filtrer avec la méthode Sobel

    Returns:
        np array: tableau numpy représentant l'image source filtrée avec la méthode Sobel
    """    
    
    # Pour le filtre de Sobel, la fonction nécessite l'import de la librarie math pour les racines carrés
    import math
    
    # Copie de l'image source
    imgDes = imgSrc.copy()
    
    # Définit le filtre à appliquer en abscisse
    filterX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    # Définit le filtre à appliquer en ordonnée
    filterY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    
    # Récupération de le nombre de pixel dans la longueur de l'image
    width = len(imgSrc)
    # Pour chaque ligne..
    for x in range(width):
        # .. on récupère la taille de la colonne
        height = len(imgSrc[x])
        # .. et pour chaque colonne..
        for y in range(height):
            
            # Initialise la somme des pixels filtrés avec le filtre des abscisses
            xSum = 0
            # Initialise la somme des pixels filtrés avec le filtre des ordonnées
            ySum = 0
            
            # On souhaite utiliser seulement la première et la dernière colonne de chaque filtre, en commançant par la dernier, d'où l'index -1 
            # Explications plus complètes ici : https://www.lama.univ-savoie.fr/mediawiki/index.php/Segmentation_d%27image_par_d%C3%A9tection_de_contours_et_algorithme_%22ligne_de_partage_des_eaux%22#Utilisation_des_masques_de_convolution
            for i in range(-1,2):
                for j in range(-1,2):
                    # Vérifie qu'on ne sort pas de l'image
                    if (0 <= x+i and x+i < width) and (0 <= y+j and y+j < height):
                        # Utilise le masque de convolution des abscisses
                        xSum += imgSrc[x+i][y+j] * filterX[i][j]
                        # Utilise le masque de convolution des ordonnées
                        ySum += imgSrc[x+i][y+j] * filterY[i][j]
            
            imgDes[x][y] = math.sqrt(xSum*xSum + ySum*ySum)
    
    return imgDes

print("Filtre Sobel\n")
img_color_cvt_gray_sobel = sobelFilter(img_color_cvt_gray)
img_color_cvt_gray_sobelOpenCV = cv2.Sobel(img_color_cvt_gray, cv2.CV_8U, 0, 1, ksize=3)

# Affiche les images
cv2.imshow("Image convertit en niveau de gris", img_color_cvt_gray)
cv2.imshow("Image convertit et filtree avec l'algorithme de Sobel", img_color_cvt_gray_sobel)
cv2.imshow("Image convertit et filtree avec fonction Sobel", img_color_cvt_gray_sobelOpenCV)

# Attente d'un entrée clavier utilisateur quelconque
cv2.waitKey(0)

# Ferme toutes les fenêtres
cv2.destroyAllWindows()