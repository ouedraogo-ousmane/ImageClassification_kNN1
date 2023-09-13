## Importation des librairies utiles

import numpy as np
import cv2

## Initialisation

labels  = ["chien","chat","panda"]
np.random.seed(0)

# Initialisation des poids W et b
W = np.random.randn(3,3072)
b = np.random.randn(3)

## Chargement de l'image et traitement

orig = cv2.imread("./../Images/animals/cats/cats_00022.jpg")
image = cv2.resize(orig,(32,32)).flatten()

scores = W.dot(image) + b

## Affichage du score selon le label

for (label,score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))
    
## Visualisation du resultat

cv2.putText(orig,"Label : {}".format(labels[np.argmax(scores)]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
cv2.imshow("Image",orig)
cv2.waitKey(0)