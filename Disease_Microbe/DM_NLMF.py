#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from Disease_Microbe.DM_train import DM_method
from Disease_Microbe.util import get_KNN, getSimilarMatrix
import math

def getL(A):
    D = np.diag(np.sum(A,axis=1))
    D_ = np.diag(np.sum(A,axis=0))
    L = (D+D_)-(A+A.transpose())
    return L


# get Gradient
def getG(P,V,Y,c,λ,α,L,U):
    m,_ = U.shape
    PV = np.matmul(P,V)
    YPV = (c-1)*(np.matmul(np.multiply(Y,P),V))
    YV = c*np.matmul(Y,V)
    I = np.eye(m)
    ILU = np.matmul(np.add(λ*I,α*L),U)
    G = np.add(np.subtract(np.add(PV,YPV),YV),ILU)
    return G

def function(Y, Sd, St, c, r, K1, λd, λt, α, β, γ,maxcnt):
    # Initialize
    m,n = Y.shape
    U = np.random.normal(loc=0.0, scale=1/math.sqrt(r), size=(m,r))
    V = np.random.normal(loc=0.0, scale=1/math.sqrt(r), size=(n,r))
    φ = np.zeros((m, r))
    Φ = np.zeros((n, r))

    A = get_KNN(Sd, K1)
    B = get_KNN(St, K1)

    Ld = getL(A)
    Lt = getL(B)
    for t in range(0, maxcnt):
        # get P
        P = np.empty((m, n))
        for i in range(m):
            for j in range(n):
                try:
                    temp = math.exp(np.dot(U[i], V[j]))
                    P[i][j] = temp / (1 + temp)
                except:
                    P[i][j] = 1
        Gd = P @ V + (c-1)*(np.multiply(Y, P)) @ V - c*Y @ V + (λd*np.eye(m)+α*Ld) @ U
        Gt = P.transpose() @ U + (c-1)*(np.multiply(Y.transpose(), P.transpose())) @ U - c*Y.transpose() @ U + (λt*np.eye(n)+β*Lt.transpose()) @ V
        φ += np.power(Gd,2)
        U -= γ*(Gd/np.sqrt(φ))
        Φ += np.power(Gt,2)
        V -= γ*(Gt/np.sqrt(Φ))

    return U, V

class DM_NRLF(DM_method):
    def __init__(self, A, lambdad=8,lambdat=4, alpha=0.03125, beta=0.0625, gama=0.03125):
        self.gama = gama
        self.beta = beta
        self.alpha = alpha
        self.lambdat = lambdat
        self.lambdad = lambdad
        super(DM_NRLF, self).__init__(A)


    def fun(self, A):
        Nd,Nm = A.shape
        A_ = A

        KM = getSimilarMatrix(A_.transpose(), 1)
        KD = getSimilarMatrix(A_, 1)

        # base on experice to modified the argument
        K1 = 5
        c = 8
        r = 18

        U,V = function(A_,KD,KM,c,r,K1,self.lambdad, self.lambdat, self.alpha, self.beta, self.gama,100)

        resultP = np.zeros((Nd,Nm))

        for i in range (0,Nd):
            for j in range (0,Nm):
                try:
                    temp = math.exp(np.dot(U[i], V[j]))
                    resultP[i][j] = temp/(1+temp)
                except:
                    resultP[i][j] = 1

        return resultP