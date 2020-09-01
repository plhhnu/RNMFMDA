#! /usr/bin/python
# -*- coding: utf-8 -*-

import random
import math
import numpy as np
from util import getSimilarMatrix

class DM_PU:
    def __init__(self, A):
        self.A = A

    def get_nr(self, per=1):
        A = self.A.copy()
        m,n = A.shape
        a = [(i, j) for i in range(m) for j in range(n) if self.A[i, j]]
        AU = [(i, j) for i in range(m) for j in range(n) if not self.A[i, j]]
        nu = int(per * len(a))
        spy = np.array(self.div_sample(a))
        A[spy[:,0], spy[:,1]] = 0
        A_ = self.fun()
        u_socre = [(A_[i][j],i,j) for (i,j) in AU]
        u_socre.sort()
        RN = [(u_socre[i][1],u_socre[i][2]) for i in range(nu)]
        return np.array(RN)

    def fun(self):
        return self.A
    
    def div_sample(self, a):
        nofp = len(a)
        nofs = nofp*0.1
        if nofs<1:
            nofs = 1
        onetest = random.sample(range(nofp),math.floor(nofs)+1)
        pt = [a[i] for i in onetest]
        return pt


    