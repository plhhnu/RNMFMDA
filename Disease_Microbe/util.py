#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i]) ** 2
    gamad =  γ_*dimensional / np.sum(sd.transpose())
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i] - IP[j])) ** 2)
    return K


def get_KNN(Matrix, K):
    ReMatrix = np.zeros(Matrix.shape)
    dimensional = Matrix.shape[0]
    Matrix_self = Matrix - np.eye(dimensional)
    for i in range(dimensional):
        a = [(Matrix_self[i,j], j)for j in range(dimensional)]
        a.sort(reverse=True)
        for j in range(K):
            e = int(a[j][1])
            ReMatrix[i, e] = 1
    return np.multiply(ReMatrix, Matrix)