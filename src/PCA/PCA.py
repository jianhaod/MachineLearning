#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: PCA.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-17
"""

from numpy import *

def loadDataSet(filename, delim = '\t'):
    
    fr = open(filename)
    stringArr = [line.strip().split() for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]

    return mat(datArr)


def pca(dataMat, topNfeat = 9999999):

    meanVals = mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar = 0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat): -1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    return lowDDataMat, reconMat

def replaceNanWithMean():
    
    datMat = loadDataSet('../../data/PCA/secom.data', ' ')
    numFeat = shape(datMat)[1]

    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        datMat[nonzero(isnan(datMat[:, i]))[0], i] = meanVal

    return datMat


if __name__ == '__main__':

    dataMat = loadDataSet('../../data/PCA/testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print lowDMat
    print reconMat
    print shape(lowDMat)

    lowDMat, reconMat = pca(dataMat, 2)
    print lowDMat
    print reconMat
    print shape(lowDMat)

    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar = 0)

    eigVals, eigVects = linalg.eig(mat(covMat))
    print eigVals

    



