#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: NaiveBayes.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-03
"""

from numpy import *
import re

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

def setOfwords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word

    return returnVec

def bagOfWordsVecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec

def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    pONum = ones(numWords)
    plNum = ones(numWords)
    pODenom = 2.0
    plDenom = 2.0 

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            plNum += trainMatrix[i]
            plDenom += sum(trainMatrix[i])
        else:
            pONum += trainMatrix[i]
            pODenom += sum(trainMatrix[i])

#    plVect = plNum/plDenom
#    pOVect = pONum/pODenom

    plVect = log(plNum/plDenom)
    pOVect = log(pONum/pODenom)

    return pOVect, plVect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString):
    
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)

    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    
    docList = []; classList = []; fullText = []

    for i in range(1,26):
        wordList = textParse(open('../../data/NaiveBayes/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('../../data/NaiveBayes/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfwords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfwords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print 'the error rate is: ', float(errorCount)/len(testSet)

# main test
if __name__ == '__main__':
    
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

    print myVocabList

    resultmp = setOfwords2Vec(myVocabList, listOPosts[0])
    print resultmp
    resultmp = setOfwords2Vec(myVocabList, listOPosts[3])
    print resultmp

    trainMat = []
    
    for postinDoc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList, postinDoc))

    pOV, PlV, pAb = trainNBO(trainMat, listClasses)
    print "pOV"
    print  pOV
    print "PlV" 
    print PlV
    print "pAb"  
    print pAb

    # test NB
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, pOV, PlV, pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, pOV, PlV, pAb)

    # cut txt content
    mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    splitresult = mySent.split()
    print splitresult
    
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(mySent)
    print listOfTokens

    splitresult = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    print splitresult

    emailText = open('../../data/NaiveBayes/email/ham/6.txt').read()
    print emailText
    listOfTokens = regEx.split(emailText)
    print listOfTokens

    spamTest()

