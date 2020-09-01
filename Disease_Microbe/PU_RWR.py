#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append("Disease_Microbe")
from util import getSimilarMatrix
from DM_PU import DM_PU
import numpy as np

class PU_RWR(DM_PU):
    def __init__(self, A, dss):
        self.A = A
        self.dss = dss

    def fun(self, sa=0.5, λ=0.9, η=0.9, r=0.5):
        A_ = self.A
        DSS = self.dss
        Nd, Nm = A_.shape

        KM = getSimilarMatrix(A_.transpose(), 1)
        KD = getSimilarMatrix(A_, 1)
        KD = sa * KD + (1 - sa) * DSS

        HMM = np.zeros((Nm, Nm))
        HMD = np.zeros((Nm, Nd))
        HDD = np.zeros((Nd, Nd))
        HDM = np.zeros((Nd, Nm))

        for j in range(Nm):
            S = np.sum(A_.transpose()[j])
            dvi = np.sum(KM[j])
            for k in range(Nm):
                if S:
                    HMM[j][k] = (1 - λ) * KM[j][k] / dvi
                else:
                    HMM[j][k] = KM[j][k] / dvi

        for j in range(Nm):
            S = np.sum(A_.transpose()[j])
            for k in range(Nd):
                if S:
                    HMD[j][k] = λ * (A_.transpose()[j][k]) / S
                else:
                    HMD[j][k] = 0

        csum = np.array(np.sum(A_.transpose(), axis=0))

        for j in range(Nd):
            dvi = np.sum(KD[j])
            for k in range(Nd):
                if csum[k]:
                    HDD[j][k] = (1 - λ) * KD[j][k] / dvi
                else:
                    HDD[j][k] = KD[j][k] / dvi

        for j in range(Nd):
            for k in range(Nm):
                if csum[j]:
                    HDM[j][k] = λ * A_[j][k] / csum[j]
                else:
                    HDM[j][k] = 0

        temp1 = np.hstack((HMM, HMD))
        temp2 = np.hstack((HDM, HDD))
        H = np.vstack((temp1, temp2))

        prst = np.zeros((Nm, Nm))
        prst[range(Nm), range(Nm)] = η
        prst = np.vstack((prst, np.full((Nd, Nm), (1 - η) / Nd)))
        pst = prst.copy()
        change = 1
        while change > 10e-12:
            new_pst = (1 - r) * np.matmul(H.transpose(), pst) + r * prst
            error = new_pst - pst
            change = np.linalg.norm(error)
            pst = new_pst.copy()

        return pst[Nm::, :]