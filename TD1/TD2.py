from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Noms des observations
ObsNamesTrain = np.array(['A','B','C','D','E','F','G','H','I','J'])

# Variables descriptives
XTrain = np.array([[8, 2],
       [4, 5],
       [4, 3],
       [7, 8],
       [2, 4],
       [7, 4],
       [1, 6],
       [8, 3],
       [6, 2],
       [5, 8]])

# Classes réelles (labels) 
YTrain = np.array([1,2,1,2,1,2,1,1,1,2])



#### Base de Test :

# Noms des observations
ObsNamesTest = np.array(['K','L','M'])

# Variables descriptives
XTest = np.array([[4, 7],
       [6, 6],
       [1, 2]])

# Classes réelles (labels) que l'on supposera inconnues 
YTest = np.array([1,2,1])
# Noms des observations
ObsNamesTrain = np.array(['A','B','C','D','E','F','G','H','I','J'])

# Variables descriptives
XTrain = np.array([[8, 2],
       [4, 5],
       [4, 3],
       [7, 8],
       [2, 4],
       [7, 4],
       [1, 6],
       [8, 3],
       [6, 2],
       [5, 8]])

# Classes réelles (labels) 
YTrain = np.array([1,2,1,2,1,2,1,1,1,2])



#### Base de Test :

# Noms des observations
ObsNamesTest = np.array(['K','L','M'])

# Variables descriptives
XTest = np.array([[4, 7],
       [6, 6],
       [1, 2]])

# Classes réelles (labels) que l'on supposera inconnues 
YTest = np.array([1,2,1])

K = 2 #Nombre de classes
q = 1 #Nombre de voisins
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(XTrain, YTrain)


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Decision Boundaries for KNN-1
xx1, xx2 = make_meshgrid(XTrain[:,0], XTrain[:,1])
Yhat = neigh.predict(np.c_[xx1.ravel(), xx2.ravel()])
Yhat = Yhat.reshape(xx1.shape)


figScatter = plt.figure(figsize=(7.5,7))
ax = figScatter.add_subplot(1,1,1)
out = ax.contourf(xx1, xx2, Yhat, alpha=0.2)
ax.scatter(XTrain[(np.where(YTrain==1)[0]),0].tolist(), XTrain[(np.where(YTrain==1)[0]),1].tolist(), color='purple', marker='*', label='Train : C1',s=150)
ax.scatter(XTrain[(np.where(YTrain==2)[0]),0].tolist(), XTrain[(np.where(YTrain==2)[0]),1].tolist(), color='orange', marker='*', label='Train : C2',s=150)
ax.scatter(XTest[:,0].tolist(), XTest[:,1].tolist(), color='blue', marker='o', label='Test samples',s=150)
ax.legend(fontsize=15, loc='upper left')
ax.scatter(XTrain[(np.where(YTrain==1)[0]),0].tolist(), XTrain[(np.where(YTrain==1)[0]),1].tolist(), color='purple', marker='*', label='Class 1',s=150)
ax.scatter(XTrain[(np.where(YTrain==2)[0]),0].tolist(), XTrain[(np.where(YTrain==2)[0]),1].tolist(), color='orange', marker='*', label='Class 2',s=150)
ax.set_xlabel("Variable 1",fontsize=16)
ax.set_ylabel("Variable 2",fontsize=16)
ax.set_xticks(np.arange(1,9,1))
ax.set_yticks(np.arange(1,9,1))
ax.set_title("Frontière de décision obtenue par les qNN avec Scikit-Learn",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.grid()
plt.show()