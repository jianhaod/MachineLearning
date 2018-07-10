#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: AdaBoost.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-10
"""

from numpy import *

def loadSimpData():

    datMat = matrix([[1., 2.1],
            [2., 1.1],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    
    retArray = ones((shape(dataMatrix)[0], 1))
    
    if threshVal == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray

def buildStump(dataArr, classLabels, D):
    
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr

                #print "split: dim: %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                #    (i, threshVal, inequal, weightedError)
                
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D: ", D.T
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, expon)
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print "total error: ", errorRate, "\n"
        
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass, classifierArr):

    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst

    return sign(aggClassEst)

def loadDataSet(filename):
    
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')

        for i in range(numFeat - 1):
            lineArr.append(float(currLine[i]))
        
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))

    return dataMat, labelMat


if __name__ == '__main__':

    datMat, classLabels = loadSimpData()
    print datMat, "\n"
    print classLabels, "\n"

    D = mat(ones((5, 1))/5)
    bestStump, minError, bestClassEst = buildStump(datMat, classLabels, D)
    print bestStump, "\n"
    print minError, "\n"
    print bestClassEst, "\n"

    adaBoostTrainDS(datMat, classLabels, 9)

    classifierArr = adaBoostTrainDS(datMat, classLabels, 30)
    
    result = adaClassify([0, 0], classifierArr)
    print result
    result = adaClassify([[5, 5], [0, 0]], classifierArr)
    print result
    
    datArr, labelArr = loadDataSet('../../data/AdaBoost/horseColicTraining2.txt')
    classifierArr = adaBoostTrainDS(datArr, labelArr, 10)

    testArr, testLabelArr = loadDataSet('../../data/AdaBoost/horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArr)
    
    print prediction10

    errArr = mat(ones((67, 1)))
    result = errArr[prediction10 != mat(testLabelArr).T].sum()
    print result



