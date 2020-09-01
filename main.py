#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.sparse import csc_matrix
from Disease_Microbe.DM_NLMF import DM_NRLF
from Disease_Microbe.PU_RWR import PU_RWR
import argparse

parser = argparse.ArgumentParser('PUNR')
parser.add_argument('--rn_scale',
                    required=False,
                    type=float,                   
                    default=1,
                    help='scale of negative sample')
parser.add_argument('--Kfold_num',
                    required=False,
                    type=int,                   
                    default=5,
                    help='K fold cross validation')
args = parser.parse_args()
        
if __name__ == "__main__":
    filename = './data/disease-microbe associationg number.prn'
    sfilename = './data/disease symptom similarity.txt'

    Nd = 39
    Nm = 292

    fp = open(filename)
    a = [(int(i[0]) - 1, int(i[1]) - 1) for i in [eline.split() for eline in fp]]
    fp.close()

    arow = [i[0] for i in a]
    acol = [i[1] for i in a]

    fp = open(sfilename)
    slist = [(int(i[0]) - 1, int(i[1]) - 1, float(i[2])) for i in [eline.split() for eline in fp]]
    fp.close()

    srow = [i[0] for i in slist]
    scol = [i[1] for i in slist]
    data = [i[2] for i in slist]

    A = np.zeros((Nd, Nm))
    A[arow, acol] = 1

    dss = csc_matrix((data, (srow, scol)), shape=(Nd, Nd)).toarray()

    x = DM_NRLF(A)
    if args.rn_scale:
        x.pu(PU_RWR(A, dss).get_nr(args.rn_scale))
    x.tarin(args.Kfold_num)
    print("AUC:", x.auc)
    

    

    