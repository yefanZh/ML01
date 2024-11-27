import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.stats as stats

np.random.seed(407)

# RÉPONSE À LA QUESTION 2.1 :

# Nous appelerons "piori_hat" le vecteur contenant l'estimation des probabilités à priori de chaque classe.

def estimation_priors(K,Y):
    '''
    Parameters
    ----------
    K : int
        Number of classes.
    Y : np.array
        Real labels.

    Returns
    -------
    piori_hat : Array of floats
        Class proportions.
    '''
    
    piori_hat = np.zeros(K)
    n = np.shape(Y)[0]
    
    for k in range(K):
        piori_hat[k] = np.sum(Y==k+1)/n
    
    return piori_hat
    
    
# RÉPONSE À LA QUESTION 2.2 :

# Pour une variable catégorielle :

def estimation_vraisemblance_cat(Xj,Y,K,T):
    '''
    Parameters
    ----------
    Xj : np.array
        Vecteur contenant les données d'apprentissage de la variable j.
    Y : np.array
        Labels des observations de la base d'apprentissage
    K : int
        Number of classes.
    T : int
        Number of discrete profiles for the categorical variable Xj.

    Returns
    -------
    pHatj : Array of floats
        Estimation des vraisemblance catégorielles dans chaque classe.
    '''
    
    pHatj = np.zeros((K,T))
    
    for k in range(K):
        nk = np.sum(Y==k+1)
        Ik = np.where(Y==k+1)
        for t in range(T):
            pHatj[k,t] = np.sum(Xj[Ik[0]]==t)/nk
    
    return pHatj
    
    
    
    
    

# Pour une variable numérique :

def estimation_vraisemblance_num(Xj,Y,K):
    '''
    Parameters
    ----------
    Xj : np.array
        Vecteur contenant les données d'apprentissage de la variable j.
    Y : np.array
        Labels des observations de la base d'apprentissage
    K : int
        Number of classes.

    Returns
    -------
    mu_j : Array of floats
        Estimation des moyennes des gaussiennes dans chaque classe.
    sigma_j : Array of floats
        Estimation des écart-types des gaussiennes dans chaque classe.
    '''

    mu_j = np.zeros(K)
    sigma_j = np.zeros(K)
    
    for k in range(K):
        Ik = np.where(Y==(k+1))
        mu_j[k] = np.mean(Xj[Ik[0]])
        sigma_j[k] = np.std(Xj[Ik[0]])
    
    return mu_j, sigma_j

K = 2
mu11 = 1/3
mu12 =1.75
sigma11 =0.5163977794943222
sigma12= 0.5

mu21 = 1.5
mu22 = 2.5
sigma21 = 0.8366600265340756
sigma22 = 0.5773502691896257

nTrain = 10
YTrain = np.array([1,2,1,2,1,2,1,1,1,2])
XTrain = np.array([[0.5,1.5,0,1,0,1,0,0,0,1],[1.5,2.5,1,2,1,2,1,1,1,2]]).T

# RÉPONSE À LA QUESTION 3.1 :

# Estimation des probabilités à priori :
priori_hat = estimation_priors(K,YTrain)

# Estimation des paramètres des vraisemblances de chaque classe pour la variable 1 qui est numérique :
XTrain_1 = XTrain[:,0]
mu_1, sigma_1 = estimation_vraisemblance_num(XTrain_1,YTrain,K)

# Estimation des paramètres des vraisemblances de chaque classe pour la variable 2 qui est numérique :
XTrain_2 = XTrain[:,1]
mu_2, sigma_2 = estimation_vraisemblance_num(XTrain_2,YTrain,K)

# Estimation des paramètres des vraisemblances de chaque classe pour la variable 3 qui est catégorielle :
XTrain_3 = XTrain[:,2]
pHat3 = estimation_vraisemblance_cat(XTrain_3,YTrain,K,T)