# Importation des librairies
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# Construction des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
args = vars(ap.parse_args())

# Lecture des images pendant le chargement
print("[INFO] chargement des images ...")
imagesPaths = list(paths.list_images(args["dataset"]))


# Traitement de chaque image
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data , labels) = sdl.loader(imagePaths=imagesPaths, verbose=500)
data = data.reshape((data.shape[0],3072))

# Encodage des données
lb = LabelEncoder()
labels = lb.fit_transform(labels)

# Partitionnement des données
(trainX , testX,trainY,testY) = train_test_split(data,labels, test_size=0.25, random_state=5)


# application de la regularization
# Vu que nous avons deux methodes de regularization
for r in (None, "l1","l2"):
    print("[INFO] entrainement du model avec une penalité de {}".format(r))
    model = SGDClassifier(loss="log_loss",penalty=r, max_iter=100, learning_rate="constant",eta0=0.01,random_state=42)
    model.fit(trainX,trainY)

    # evaluation de la precision
    acc = model.score(testX, testY)
    print("[INFO] {} score de la penalité : {:.2f}%".format(r,acc * 100))
    
    
