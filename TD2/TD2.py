import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.stats as stats

## Permet la reproducibilité des résultats
np.random.seed(10)

mu1 = 37.8
mu2 = 39
sigma1 = 0.4
sigma2 = 0.3


n1 = 80
n2 = 20
n = n1 + n2
p = 0.2

y = np.zeros(n)
X = np.zeros((n,1))

for i in range(0,n1):
    y[i] = 1
    X[i] = np.random.normal(mu1, sigma1, 1)
for i in range(n1,n):
    y[i] = 2
    X[i] = np.random.normal(mu2, sigma2, 1)

X = np.vstack([np.random.normal(mu1,sigma1,(n1,1)),np.random.normal(mu2,sigma2,(n2,1))])
y = np.concatenate([np.full(n1,1), np.full(n2,2)])


def Plot_densities(mu1, sigma1, mu2, sigma2, X, y):
    '''
    Trace les densités des lois normales N(mu1, sigma1) et N(mu2, sigma2) ainsi que les 
    échantillons X en fonction de leur classes y
    :param mu1: moyenne de la loi 1
    :param sigma1: écart-type de la loi 1
    :param mu2: moyenne de la loi 2
    :param sigma2: écart-type de la loi 2
    :param X: 2d array contenant les m échantillons
    :param y: 1d array contenant les classes associées aux m échantillons
    :return: None
    '''
    figScatter = plt.figure(figsize=(10,7))
    ax = figScatter.add_subplot(1,1,1)
    xx = np.linspace(mu1 - 4*sigma1, mu2 + 4*sigma2, 100)
    #Tracé des densité théoriques
    density1 = stats.norm.pdf(xx, mu1, sigma1)
    density2 = stats.norm.pdf(xx, mu2, sigma2)
    ax.plot(xx, density1, color='deepskyblue', linewidth = 1.5)
    ax.plot(xx, density2, color='red', linewidth = 1.5)
    ax.axhline(0,color='black')
    #Tracé des échantillons par classe
    ax.scatter(X[np.where(y==1),0].tolist(), [0]*n1, color='deepskyblue', marker='.', label='Class 1',s=200)
    ax.scatter(X[np.where(y==2),0].tolist(), [0]*n2, color='red', marker='.', label='Class 2',s=200)
    #Legendes
    ax.legend(fontsize=20, loc='upper left')
    ax.set_xlabel("Température",fontsize=20)
    ax.set_ylabel("Fonction de densité",fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set(xlim=(np.min(xx), np.max(xx)))
    ax.set(ylim=(-0.1, np.maximum(np.max(density1),np.max(density2))+0.1))
    plt.show()


def BayesRule_theoretical(mu1, mu2, sigma1, sigma2):
    ## On utilise une "factory" pour fabriquer un classifieur X -> {1,2}
    def BRT(xi) :
        f1xi =  1/(sigma1*np.sqrt(2*np.pi)) * np.exp(-(xi - mu1)**2/(2*sigma1**2))
        f2xi =  1/(sigma2*np.sqrt(2*np.pi)) * np.exp(-(xi - mu2)**2/(2*sigma2**2))
        fXx = (1-p)*f1xi + p*f2xi
        PY1xi = (1-p)*f1xi/fXx
        PY2xi = p*f2xi/fXx
        yhat = np.argmax([PY1xi,PY2xi])+1
        return yhat
    return BRT

# Diagnostic pour chaque patient à partir du classifieur de Bayes Théorique :
BayesRule_th = BayesRule_theoretical(mu1, mu2, sigma1, sigma2)
y_hat = np.zeros(n)
for i in range(0,n):
    y_hat[i] = BayesRule_th(X[i,:])
# Calculs des taux d'erreurs :
errGlobal = np.sum(y!= y_hat)/n
print('Taux erreur global du classifieur de Bayes théorique =', errGlobal)

err_k = np.zeros((1,2))
for k in range(0,2):
    nk = np.sum(y==k+1)
    Ik = np.where(y == (k + 1))
    print(Ik)
    err_k[0,k] = sum(y_hat[Ik[0]]!=k+1)/nk
print('Taux erreurs par classe du classifieur de Bayes théorique =', err_k[0])

# RÉPONSE À LA QUESTION 3.2:

I1 = np.where(y == 1)
mu1hat = np.mean(X[I1[0],0])
sigma1hat = np.std(X[I1[0],0])

I2 = np.where(y == 2)
mu2hat = np.mean(X[I2[0],0])
sigma2hat = np.std(X[I2[0],0])

p_hat = np.sum(y==2)/n
    
    
def BayesRule_empirique(mu1hat, mu2hat, sigma1hat, sigma2hat):
    ## On utilise une "factory" pour fabriquer un classifieur X -> {1,2}
    def BRT_emp(xi) :
        f1xi =  1/(sigma1hat*np.sqrt(2*np.pi)) * np.exp(-(xi - mu1hat)**2/(2*sigma1hat**2))
        f2xi =  1/(sigma2hat*np.sqrt(2*np.pi)) * np.exp(-(xi - mu2hat)**2/(2*sigma2hat**2))
        fXx = (1-p_hat)*f1xi + p_hat*f2xi
        PY1xi = (1-p_hat)*f1xi/fXx
        PY2xi = p_hat*f2xi/fXx
        yhat = np.argmax([PY1xi,PY2xi])+1
        return yhat
    return BRT_emp


# Diagnostic pour chaque patient à partir du classifieur de Bayes Empirique :
BayesRule_emp = BayesRule_empirique(mu1hat, mu2hat, sigma1hat, sigma2hat)
Yhat = np.zeros((n,1))
for i in range(0,n):
    Yhat[i,0] = BayesRule_emp(X[i,0])
    

# Calculs des taux d'erreurs :
errGlobal = np.sum(y!=Yhat.ravel())/n
print('Taux erreur global du classifieur de Bayes empirique =', errGlobal)

err_k = np.zeros((1,2))
for k in range(0,2):
    nk = np.sum(y==k+1)
    Ik = np.where(y == (k + 1))
    err_k[0,k] = sum(Yhat[Ik[0]]!=k+1)/nk
print('Taux erreurs par classe du classifieur de Bayes empirique =', err_k[0])


# On observe un surraprentissage par rapport au classifieur de Bayes théorique

m1Test = 800
m2Test = 200
mTest = m1Test + m2Test

Ytest = np.zeros((mTest,1))
Xtest = np.zeros((mTest,1))

for i in range(0,m2Test):
    Ytest[i] = 1
    Xtest[i] = np.random.normal(mu1, sigma1, 1)
for i in range(m2Test,mTest):
    Ytest[i] = 2
    Xtest[i] = np.random.normal(mu2, sigma2, 1)


# Diagnostic pour chaque patient à partir du classifieur de Bayes Théorique :
YhatTestBayes = np.zeros((mTest,1))
YhatTestEmpBayes = np.zeros((mTest,1))
for i in range(0,mTest):
    YhatTestBayes[i,0] = BayesRule_th(Xtest[i,0])
    YhatTestEmpBayes[i,0] = BayesRule_emp(Xtest[i,0])
    

# Calculs des taux d'erreurs :
errGlobalBayes = np.sum(Ytest!=YhatTestBayes)/mTest
errGlobalEmp = np.sum(Ytest!=YhatTestEmpBayes)/mTest
print('Taux erreur global du classifieur de Bayes Théorique =', errGlobalBayes)
print('Taux erreur global du classifieur de Bayes Empirique =', errGlobalEmp)

err_k_Bayes = np.zeros((1,2))
err_k_Bayes_Emp = np.zeros((1,2))
for k in range(0,2):
    mk = np.sum(Ytest==k+1)
    Ik = np.where(Ytest == (k + 1))
    err_k_Bayes[0,k] = sum(YhatTestBayes[Ik[0]]!=k+1)/mk
    err_k_Bayes_Emp[0,k] = sum(YhatTestEmpBayes[Ik[0]]!=k+1)/mk
print('Taux erreurs par classe du classifieur de Bayes Théorique =', err_k_Bayes[0])
print('Taux erreurs par classe du classifieur de Bayes Empirique =', err_k_Bayes_Emp[0])