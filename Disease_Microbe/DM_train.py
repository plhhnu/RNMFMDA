#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix

class DM_method:

    def __init__(self, interaction):
        self.A = interaction
        self.positive_addr = np.array(np.nonzero(interaction)).T
        self.zero_addr = np.array([(i, j) for i in range(interaction.shape[0]) for j in range(interaction.shape[1]) if interaction[i, j] == 0])
        self.negative_addr = None
        self.positive_sample_size = int(self.positive_addr.size/2)
        self.zero_sample_size = int(self.zero_addr.size/2)
        self.negative_sample_size = 0
        # self.SSD = Dsim
        self.label = []
        self.score = []
        self.K = 0
        self.valid_sample = None

    @property
    def is_get_result(self):
        if self.label and self.score:
            return True
        return False

    @property
    def auc(self):
        return self.performance(roc_curve)

    @property
    def aupr(self):
        return self.performance(precision_recall_curve, True)

    @property
    def get_label(self):
        if self.is_get_result:
            return self.label
        return None

    @property
    def get_score(self):
        if self.is_get_result:
            return self.score
        return None

    def performance(self, fun, ex=False):
        if not self.is_get_result:
            return None
        auct = 0
        for i in range(self.K):
            fpr, tpr, _ = fun(self.label[i], self.score[i])
            if ex:
                auct += auc(tpr, fpr)
            else:
                auct += auc(fpr, tpr)
        return auct/self.K

    def Kfoldcrossclassify(self, sample, K, fun="cv3"):
        r = []
        if fun != "cv3":
            m = np.mat(sample)
            if fun == "cv1":
                t = 0
            else:
                t = 1
            mt = self.Kfoldcrossclassify(np.array(range(np.max(m[:, t]) + 1)), K)
            # for i in range(K):
            r = [[j for j in sample if j[t] in mt[i]] for i in range(K)]
            return r

        l = sample.shape[0]
        t = sample.copy()
        n = math.floor(l / K)
        retain = l - n*K
        for i in range(K - 1):
            nt = n
            e = len(t)
            # if e % n and e % K:
            if retain > i:
                nt += 1
            a = random.sample(range(e), nt)
            r.append([t[i] for i in a])
            t = [t[i] for i in range(e) if (i not in a)]
        r.append(t)
        return r

    def prepare(self, K=0, cvt="cv3"):
        if K:
            self.K = K
            self.valid_sample = self.Kfoldcrossclassify(self.positive_addr, K, cvt)
        else:
            self.K = self.positive_addr
            self.valid_sample = np.array([np.array([i]) for i in self.positive_addr])

    def fun(self, A):
        return A

    def tarin(self, K=0, cvt="cv3"):
        self.label = []
        self.score = []
        self.prepare(K, cvt)
        for i in range(self.K):
            test = np.array(self.valid_sample[i])
            A = self.A.copy()
            A[test[:, 0], test[:, 1]] = 0
            if self.negative_sample_size:
                A[self.negative_addr[:,0], self.negative_addr[:,1]] = -1
            self.label.append(np.array([1] * test.shape[0] + [0] * self.zero_sample_size))
            A = self.fun(A)
            sco_addr = np.vstack((test, self.zero_addr))
            self.score.append(A[sco_addr[:, 0], sco_addr[:, 1]])

    def pu(self, rn):
        self.negative_addr = rn
        self.negative_sample_size = self.negative_addr.shape[0]
        self.zero_addr = np.array([i for i in self.zero_addr for j in self.negative_addr if i not in j])
        self.zero_sample_size = self.zero_addr.shape[0]

    def acc_threshold(self,label, score):
        _, _, thresholds = roc_curve(label, score)
        acc_t = 0
        precision_arg = 0
        recall_arg = 0
        specificity_arg = 0
        l = len(thresholds)
        print("{}".format(l))
        for i in thresholds:
            pre = score >= i
            a = confusion_matrix(label, pre, labels=[0, 1])
            TP = a[1][1]
            FP = a[0][1]
            FN = a[1][0]
            TN = a[0][0]
            try:
                precision_arg += TP / (TP + FP)
            except ZeroDivisionError:
                pass
            recall_arg += TP / (TP + FN)
            specificity_arg += TN / (TN + FP)
            acc_t += (TP+TN)/(TP+TN+FP+FN)
        return acc_t/l, precision_arg/l, recall_arg/l, specificity_arg/l

    @property
    def evoluate(self):
        if not self.is_get_result:
            return None
        ev = np.zeros(5)
        for i in range(self.K):
            ev += np.array(self.acc_threshold(self.label[i], self.score[i]))
        return ev/self.K

    def predict(self):
        pre = self.fun(self.A)
        t = pre[self.zero_addr[:, 0], self.zero_addr[:, 1]]
        r = [(t[i], self.zero_addr[i][0], self.zero_addr[i][1]) for i in range(self.zero_sample_size)]
        r.sort(reverse=True)
        return r
