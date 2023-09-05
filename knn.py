# Importation des librairies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

# Les arguments

ap = argparse.ArgumentParser()

ap.add_argument("-d","--dataset",required=True,help="Chemin d'accès à l'ensemble des données")
ap.add_argument("-k","--neighbors", type=int, default=1, help="Nombre de voisin proches")
ap.add_argument("-j","--jobs", type=int,default=-1, help="Nombre de jobs à executer parallèlement")

args = vars(ap.parse_args())

### Recuperation des images
print("[INFO] chargement des images")
chemin = list(paths.list_images(args["dataset"]))

## Chargement des données et preprocessing

preprocessor = SimplePreprocessor(32,32)
charger = SimpleDatasetLoader(preprocessors=[preprocessor])

(data,labels) = charger.loader(imagePaths=chemin, verbose=500)

data = data.reshape((data.shape[0],3072))

# Details sur la consommation memoire

print("[INFO] features matrix : {: .1f}MB".format(data.nbytes/(1024 * 1000.0)))

# Encodage

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Division des données 

(trainX, testX , trainY, testY) = train_test_split(data,labels,test_size=0.25, random_state=42)

# Creation du model et test
print("[INFO] creation du KNN ....")

model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX,trainY)

## Affichage du score de notre modèle
print(classification_report(testY,model.predict(testX),target_names=encoder.classes_))