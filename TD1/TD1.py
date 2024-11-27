import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def function_qNN(XTrain, YTrain, XTest, K, q):
    
       dis = np.zeros((XTrain.shape[0], XTest.shape[0]))
       
       for i in range(XTrain.shape[0]):
              for j in range(XTest.shape[0]):
                     dis[i,j] = np.linalg.norm(XTrain[i] - XTest[j])
       print("dis : \n", dis)
       dis_argsort = np.argsort(dis, axis=0)
       print("dis_argsort : \n", dis_argsort)
       result = np.zeros(XTest.shape[0])
       for j in range(XTest.shape[0]):
              pred = np.zeros(q)
              for i in range(q):
                     pred[i] = pred[i] + YTrain[dis_argsort[i,j]]
              print("pred j : \n", pred)
              pred_count = np.bincount(pred.astype(int))
              result[j] = np.argmax(pred_count)
       return result

#### Base d'apprentissage :

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

#function_qNN(XTrain, YTrain, XTest, 2, 1)
pred = function_qNN(XTrain, YTrain, XTest, 2, 1)

print("pred_result : \n", pred)

