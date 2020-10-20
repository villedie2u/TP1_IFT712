# -*- coding: utf-8 -*-

#####
# Vos Noms (VosMatricules) .~= À MODIFIER =~.
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # AJOUTER CODE ICI
            
            #Taille N de x_train et t_train
            N=len(t_train)
            
            #Calcul de p
            p=0
            N2=0
            for t in t_train:
                p=p+t
                if t==0:
                    N2=N2+1.0
            N1=p
            
            #Calcul de mu1 et mu2
            mu1=0
            mu2=0
            for i in range(N):
                mu1=mu1+(t_train[i]*x_train[i])
                mu2=mu2+((1-t_train[i])*x_train[i])
            mu1=mu1/N1
            mu2=mu2/N2
            
            #Calcul de sigma
            S=np.zeros((2,2))
            S1=np.zeros((2,2))
            S2=np.zeros((2,2))
            mu1_matrice = np.array([[mu1[0],mu1[1]]])
            mu2_matrice = np.array([[mu2[0],mu2[1]]])
            for i in range(N):
                xi = np.array([[x_train[i][0],x_train[i][1]]])
                #terme = 0
                #print(xi)
                if t_train[i]==1:
                    terme = np.dot((xi-mu1_matrice).transpose(),(xi-mu1_matrice))
                    #print(terme)
                    S1=S1+terme
                elif t_train[i]==0:
                    terme = np.dot((xi-mu2_matrice).transpose(),(xi-mu2_matrice))
                    #print(terme)
                    S2=S2+terme
            S1=S1/N1
            S2=S2/N2
            
            S=((N1/N)*S1)+((N2/N)*S2)
            
            lambda_matrice = self.lamb*np.identity(2)
            S=S+lambda_matrice
            
            
            #Calcul de self.w
            mu1_moins_mu2 = mu1_matrice-mu2_matrice
            S_inv = np.linalg.inv(S)
            self.w = np.dot(S_inv,mu1_moins_mu2.transpose())
            
            #Calcul de self.w_0
            PC1=N1/N
            PC2=N2/N
            terme1= -0.5*np.dot(np.dot(mu1,S_inv),(mu1.transpose()))
            terme2= 0.5*np.dot(np.dot(mu2,S_inv),(mu2.transpose()))
            terme3= np.log(PC1/PC2)
            
            self.w_0 = terme1 + terme2 + terme3
            
            

        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            # AJOUTER CODE ICI

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            # AJOUTER CODE ICI

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        # AJOUTER CODE ICI
        result = 0
        pred = self.w_0 + self.w[0]*x[0] + self.w[1]*x[1]
        
        if (pred>0):
            result = 1
        
        return result

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # AJOUTER CODE ICI
        
        erreur = 1
        if (t == prediction):
            erreur = 0
        
        return erreur

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w