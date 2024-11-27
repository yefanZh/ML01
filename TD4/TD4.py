import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.stats as stats

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def plot_2d_data(XTrain, YTrain):
    # Créer la figure et les axes
    plt.figure(figsize=(10, 6))

    # Tracer les points de la classe 1
    plt.scatter(XTrain[YTrain == 1, 0], XTrain[YTrain == 1, 1], color='blue', label='Classe 1')

    # Tracer les points de la classe 2
    plt.scatter(XTrain[YTrain == 2, 0], XTrain[YTrain == 2, 1], color='red', label='Classe 2')

    # Ajouter des légendes, des titres et des labels aux axes
    plt.title('Représentation des nuages de points pour les classes 1 et 2')
    plt.xlabel('Variable descriptive 1')
    plt.ylabel('Variable descriptive 2')
    plt.legend()

    # Afficher le graphique
    plt.show()


np.random.seed(407)
rng = np.random.default_rng()
XTrain1 = rng.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 3]], size=500)
YTrain1 = np.ones(500)

XTrain2 = rng.multivariate_normal(mean=[0, 4], cov=[[2, 1], [1, 3]], size=500)
YTrain2 = np.ones(500) + 1

XTrain = np.concatenate((XTrain1, XTrain2), axis=0)
YTrain = np.concatenate((YTrain1, YTrain2), axis=0)

# plot_2d_data(XTrain, YTrain)

# Créer un classifieur naïf de Bayes
gnb = GaussianNB()
y_pred = gnb.fit(XTrain, YTrain).predict(XTrain)
print("Nombre d'erreurs de classification :", (YTrain != y_pred).sum())

clf = QuadraticDiscriminantAnalysis()
clf.fit(XTrain, YTrain)
y_pred = clf.predict(XTrain)
print("Nombre d'erreurs de classification :", (YTrain != y_pred).sum())

lda = LinearDiscriminantAnalysis()
lda.fit(XTrain, YTrain)

