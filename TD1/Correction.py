import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def function_qNN(XTrain, YTrain, XTest, K, q):
    
    # Recherche des q plus proches voisins de l'observation test :
    n = XTrain.shape[0] #nombre d'observations dans la base d'apprentissage
    distXtest = np.zeros((1, n))
    for i in range(0, n):
        distXtest[0,i] = np.linalg.norm(XTest - XTrain[i,:]) #calcul de la distance euclidienne entre l'observation test et chaque observation d'apprentissage
    print("distXtest : \n", distXtest)
    ArgsortDist = np.argsort(distXtest) 
    qNN_indix = ArgsortDist[0,0:q] #indices des q plus proches voisins
    
    # Estimation de la classe de l'observation test :
    score_class = np.zeros(K)
    for i in range(0, q):
        score_class[YTrain[qNN_indix[i]]-1] = score_class[YTrain[qNN_indix[i]]-1] + 1
    Yhat = np.argmax(score_class)+1 # Yhat correspond à la classe estimer de l'observation test
    
    return Yhat

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
for i in range(0, XTest.shape[0]):
    xtest_i = XTest[i,:]
    Yhat_i = function_qNN(XTrain, YTrain, xtest_i, K, q)
    print('Estimated class of point ', ObsNamesTest[i], ' = ', Yhat_i)

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


xx1, xx2 = make_meshgrid(XTrain[:,0], XTrain[:,1])
XXgrid = np.c_[xx1.ravel(), xx2.ravel()]
Z = np.zeros((1, XXgrid.shape[0]))
for i in range(0, XXgrid.shape[0]):
    Z[0,i] = function_qNN(XTrain, YTrain, XXgrid[i,:], K, q)
Z = Z.reshape(xx1.shape)


figScatter = plt.figure(figsize=(7.5,7))
ax = figScatter.add_subplot(1,1,1)
out = ax.contourf(xx1, xx2, Z, alpha=0.2)
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
ax.set_title("Frontière de décision obtenue par le classifieur qNN",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.grid()

plt.show()