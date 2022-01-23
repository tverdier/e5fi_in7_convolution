# -*- coding: utf-8 -*-

"""
TP KNN
"""

# import subprocess

# print("Downloading the dataset..")
# gdownResult = subprocess.run(["gdown", "--id", "1sQEIPh3bdKQ_1J3g0Z8CRqD6uU7v746l"], stderr=subprocess.PIPE, text=True)
# print(gdownResult)

# print("Unzip the dataset..")
# unzipResult = subprocess.run(["unzip", "-u", "dataset.zip"], stderr=subprocess.PIPE, text=True)
# print(unzipResult)

# Librairie permettant de représenter le système de fichiers sous forme d'objet
import pathlib
# Librarie OpenCV
import cv2
# Librarie numpy
import numpy as np
# Librairie permettant d'utiliser des fonctions dépendantes du sytème d'exploitation
import os
# Librairie permettant notamment de générer des nombres aléatoires
import random
# Librairie permettant notamment d'afficher proprement des images, graphiques, etc..
import matplotlib.pyplot as plt

        
# Define Feature Extractor
    
## Raw pixel values
def image_to_feature_vector(image, size=(32, 32)):
    """
    Redimensionne l'image en argument en 32x23 et construit un tableau d'1D avec l'image
    Cela permet d'utiliser le résultat comme feature d'une modèle de ML

    Args:
        image (Mat): image lue avec cv2.imread
        size (tuple, optional): Taille de l'image après avoir été redimensionnée. Valeur par défaut : (32, 32).

    Returns:
        Mat: Image redimensionnée et "applatie"/transformée en tableau 1D
    """    
    return cv2.resize(image, size).flatten()

## Histogramme
# import imutils
def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Renvoie l'histogramme "applati"/transformé en tableau 1D de l'image en argument
    Cela permet d'utiliser le résultat comme feature d'une modèle de ML

    Args:
        image ([Mat]): image lue avec cv2.imread
        bins (tuple, optional): Nombre de subdivisions dans chaque dimensions. Valeur par défaut : (8, 8, 8).

    Returns:
        [Mat]: histogramme "applati"/transformé en tableau 1D de l'image en argument
    """
    # Changement d'espace de couleur : BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calcul de l'histogramme
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    # Normalisation de l'histogramme
    cv2.normalize(hist,hist)
    # Transformation de l'histogramme en tableau 1D
    return hist.flatten()


# Construction d'un objet associé au dossier flowers dans le répertoire courant
data_dir = pathlib.Path("./flowers")

# Initialisation du dataset
dataset = []

# Vérifie si le chemin existe
if data_dir.exists():
    print("Le dossier flowers existe.")
    # Compte le nombre d'image qu'il y a dans les sous dossiers de flowers
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Il y a", image_count, "images dans les sous-dossiers")
    
    # Initialisation du compte des images utilisées dans le dataset
    count = 0
    
    print("Construction du dataset..")
    # Pour chaque sous-dossier dans flowers
    for label in os.listdir("./flowers/"):
        # Pour chaque fichier dans le sous-dossier
        for filename in os.listdir(os.path.join("./flowers/",label)):
            # Incrémente le nombre d'image utilisées dans le dataset
            count = count + 1
            # Indique que l'on lit l'image <filename>
            # print(str(count) + " ---loading " + filename)
            # Lecture de l'image
            image = cv2.imread(os.path.join("./flowers/", label, filename))
            # Ajout au dataset de l'image associée à son dossier
            dataset.append((image, label))
            
    # Affiche 10 échantillons du dataset tirés au hasard
    # print(random.sample(dataset, 10))
    
    print("Mélange aléatoirement le dataset..")
    random.shuffle(dataset)
    
    # Renvoie un objet Figure, fig, et un tableau d'Axes
    # fig correspond au conteneur de tous les éléments
    # axs correspond à un tableau représentant les sous-parties de la figure
    # Ici la Figure sera divisé en 9 cases (9 Axes)
    fig, axs = plt.subplots(3, 3, figsize = (12, 12))
    
    # Modifie le mappage des couleurs des plots en gris
    plt.gray()

    # Description de dataset
    #   dataset[i] = (matrice, dossier)
    #   dataset[i][0] = matrice
    #   dataset[i][1] = dossier
    #   dataset[i][0][j] = ligne j de la matrice
    #   dataset[i][0][j][k] = pixel aux coordonnées [j;k] de la matrice
    for i, ax in enumerate(axs.flat):
        # Place dans Axe dans la matrice de l'image i
        ax.imshow(dataset[i][0])
        # Désactive les axes
        ax.axis('off')
        # Ajoute un titre, celui du dossier auquel appartient l'image
        ax.set_title(dataset[i][1])
    
    print("Affiche les 9 premières images du dataset")
    plt.show()

    print("Construction du modèle")
    # Définition d'extracteurs de features (variables caractéristiques)

    print("Initialisation")
    # Initialisation des paramètres de test du modèle
    rawImages = []
    features = []
    labels = []

    # Définition d'un dictionnaire permettant d'associer une espèce de fleur à un nombre
    labels_classes_mapping = {"daisy":0,"dandelion":1,"rose":2,"sunflower":3,"tulip":4}

    print("Séparation des features et des labels")
    for (image, label) in dataset:
        rawImages.append(image_to_feature_vector(image))
        features.append(extract_color_histogram(image))
        labels.append(labels_classes_mapping[label])

    print("Mise en forme et vérification")
    # Transformation des tableaux en tableau numpy
    rawImages = np.array(rawImages)
    features = np.array(features) 
    labels = np.array(labels)

    # Affiche la forme des tableau (nbCases1D, nbCases2D, ..., nbCasesND)
    print(rawImages.shape)
    print(features.shape)
    print(labels.shape)

    print("Séparation de la base de test et la base d'entraînement")

    # train_test_split diviser notre ensemble de données en sous-ensembles qui minimisera le risque de biais dans notre processus d'évaluation et de validation.
    from sklearn.model_selection import train_test_split

    print("Création de données d'entraînement avec les images bruts redimensionnées..")
    (trainRawX, testRawX, trainRawY, testRawY) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
    print("Création de données d'entraînement avec l'histogramme de l'image..")
    (trainFeatX, testFeatX, trainFeatY, testFeatY) = train_test_split(features, labels, test_size=0.25, random_state=42)

    print("Utilisation d'un modèle KNN")
    
    # KNeighborsClassifier permet de classifier les données dans un modèle à l'aide de l'algorithme des k plus proches voisins
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    print("Entraînement du modèle avec les images bruts redimensionnées en 32x32..")
    model.fit(trainRawX, trainRawY)
    acc = model.score(testRawX, testRawY)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))

    model = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    print("Entraînement du modèle avec l'histogramme des images..")
    model.fit(trainFeatX, trainFeatY)
    acc = model.score(testFeatX, testFeatY)
    print("histogram accuracy: {:.2f}%".format(acc * 100))