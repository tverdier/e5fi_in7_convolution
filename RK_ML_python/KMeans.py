# -*- coding: utf-8 -*-

"""
TP KMeans
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
# TODO
from PIL import Image
# Librairie permettant notamment de générer des nombres aléatoires
import random
# Librairie permettant notamment d'afficher proprement des images, graphiques, etc..
import matplotlib.pyplot as plt

# Construction d'un objet associé au dossier flowers dans le répertoire courant
data_dir = pathlib.Path("./flowers")

# Initialisation du dataset
dataset = []

# Vérifie si le chemin existe
if data_dir.exists():
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
            image = Image.open(os.path.join("./flowers/",label,filename))
            image.load()
            image = np.asarray(image, dtype="float32")
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
        ax.imshow(dataset[i][0].astype("int32"))
        # Désactive les axes
        ax.axis('off')
        # Ajoute un titre, celui du dossier auquel appartient l'image
        ax.set_title(dataset[i][1])
    
    print("Affiche les 9 premières images du dataset")
    plt.show()
    
    # Définition d'une taille pour redimensionner les images du dataset
    image_size = (32, 32)
    
    # Initialisation du tableau des images redimensionnées
    resized = []
    
    print("Redimensionne les images du dataset en 32x32..")
    # Redimensionne les images du dataset et insères ces dernières dans le tableau resized
    for (image, label) in dataset:
        resized.append((cv2.resize(image, image_size), label))
    
    # On considère maintenant le dataset comme étant les images redimensionnées
    dataset = resized
    
    print("Construction d'un tableau avec  seulement les images en 32x32..")
    # Création d'un tableau images qui contient seulement les images du dataset
    images = np.asarray([d[0] for d in dataset])
    # Vérifie la forme du tableau (nbCases1D, nbCases2D, ..., nbCasesND)
    print(images.shape)
    
    print("Transformation des images 32x32 en features..")
    # Modifie la forme du tableau pour les images soient "applaties"/transformées en un tableau 1D
    images = images.reshape(len(images), -1)
    # Vérifie la forme du tableau après modification (nbCases1D, nbCases2D, ..., nbCasesND)
    print(images.shape)
    
    print("Réencodage en float32..")
    # Réencodage des valeurs de chaque image en float32 et "normalisation" des valeurs entre 0 et 255
    images = np.asarray(images, dtype=np.float32) / 255.
    
    # Import d'un objet utilisé comme modèle de classification k-means
    from sklearn.cluster import MiniBatchKMeans
    
    print("Création du modèle K-Means..")
    # Indique le nombre de classes à classifier
    number_clusters = 5
    # Définition d'un modèle K-Means
    kmeans = MiniBatchKMeans(n_clusters=number_clusters)
    
    print("Entraînement du modèle K-Means..")
    # Entraînement du modèle
    if not images.all() == None:
        kmeans.fit(images)
    
    print(kmeans.labels_)
    print(kmeans.n_clusters)
    
    def map_clusters_classes(kmeans, classes):
        print("Mapping..")
        mapping = {}
        
        classes_unique = np.unique(classes)
        print("n_clusters :", kmeans.n_clusters)
        for i in range(0, kmeans.n_clusters):
            print("i=", i, " labels[i]=", classes_unique[i])
            mapping[i] = classes_unique[i]
        return mapping

    def inference(kmeans, images, classes):
        print("Inference..")
        
        mapping = map_clusters_classes(kmeans, classes)
        print("mapping : ", mapping)
        
        # Renvoie les labels auquels sont associés chaque images du dataset
        clusters = kmeans.predict(images)
        print("clusters : ", clusters)
        
        # Définition des prédictions
        predicted_classes = np.zeros(len(clusters)).astype(np.uint8)
        
        for i in range(len(clusters)):
            predicted_classes[i] = mapping[clusters[i]]
            
        print("predictions : ", predicted_classes)
        return predicted_classes
    
    labels_classes_mapping = {"daisy":0,"dandelion":1,"rose":2,"sunflower":3,"tulip":4}
    
    classes = [labels_classes_mapping[d[1]] for d in dataset]
    classes = np.asarray(classes)
    predicted_classes = inference(kmeans, images, classes)
    
    print(predicted_classes[:20])
    print(classes[:20])
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(classes, predicted_classes)
    print('Accuracy: {}\n'.format(acc))
    
    number_clusters = [5, 10, 16, 36, 64, 144, 256, 1024, 2048, 4098]
    acc_list = []

    for i in number_clusters:
        #... TODO
        kmeans = MiniBatchKMeans(n_clusters=i)
        kmeans.fit(images)
        predicted_classes = inference(kmeans, images, classes)
        acc = accuracy_score(classes, predicted_classes)
        acc_list.append(acc)
        print('Accuracy: {}\n'.format(acc))
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(number_clusters, acc_list, label='accuracy', marker='^')
    ax.legend(loc='best')
    ax.grid('on')
    ax.set_title('Accuracy per each cluster number')
    plt.show()
