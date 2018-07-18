#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: SVD.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-18
"""

from numpy import *
from numpy import linalg as la

def loadExData():

    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def euclidSim(inA, inB):
    
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):

    if len(inA) < 3:
        return 1.0

    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):

    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)

def standEst(dataMat, user, simMeas, item):
    
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
    
        if userRating == 0:
            continue
        
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        simTotal += similarity * userRating
    
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):

    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    
    if len(unratedItems) == 0:
        return 'you rated everything'

    itemScores = []

    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    return sorted(itemScores, key = lambda jj: jj[1], reverse = True)[:N]


if __name__ == '__main__':

    U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
    print U
    print Sigma
    print VT

    Data = loadExData()
    U, Sigma, VT = linalg.svd(Data)
    print Sigma

    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print Sig3

    result = U[:, :3] * Sig3 * VT[:3, :]
    print result
    
    myMat = mat(Data)
    result = euclidSim(myMat[:, 0], myMat[:, 4])
    print result
    result = euclidSim(myMat[:, 0], myMat[:, 0])
    print result

    result = cosSim(myMat[:, 0], myMat[:, 4])
    print result
    result = cosSim(myMat[:, 0], myMat[:, 0])
    print result

    result = pearsSim(myMat[:, 0], myMat[:, 4])
    print result
    result = pearsSim(myMat[:, 0], myMat[:, 0])
    print result
    
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print myMat
    
    recommend(myMat, 2)

