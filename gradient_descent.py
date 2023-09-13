# Importation des différents packages
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Fonction d'activation : Fonction sigmoïde

def sigmoid_activation(x):
    """
    Cette fonction représente la fonction d'activation sigmoïde

    Args:
        x (array-like): Les valeurs d'entrée.

    Returns:
        array-like: Les valeurs après l'application de la sigmoïde.
    """
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

# Les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning rate")
args = vars(ap.parse_args())

# Génération des données
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# Transformation de X en matrice
X = np.c_[X, np.ones((X.shape[0]))]


# Division des données en deux
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialisation de nos poids W et des pertes
W = np.random.randn(X.shape[1], 1)  # Utilisez la bonne dimension ici

losses = []

# Parcours selon les epochs
for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)
    # print(W)
    gradient = trainX.T.dot(error)
    W += -args["alpha"] * gradient
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# Évaluation du modèle
preds = predict(testX, W)
print(classification_report(preds, testY))

# Affichage graphique
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# Construction de la figure
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel('Epoch #')
plt.ylabel("Loss")
plt.show()
