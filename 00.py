# -*- coding: cp936 -*-

from numpy import *
import operator
from os import listdir


# �������ݼ�
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# ��һ��kNN������  inX-�������� dataSet-��������  labels-��ǩ k-�ڽ���k������
def classify0(inX, dataSet, labels, k):
    # ���ݼ���С
    dataSetSize = dataSet.shape[0]
    # �������
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # �������㣬�����õ���ŷ�Ͼ���
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # ����������
    sortedDistIndicies = distances.argsort()  # ������ֵ��С���������ֵ
    # ͳ��ǰk�������������
    classCount = {}
    # ѡ�������С��k����
    for i in range(k):
        # ����������������ֵ���ؿ�����ǰk����ǩ
        voteIlabel = labels[sortedDistIndicies[i]]
        # ������ǩ����Ƶ��
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # ����
    # sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # ����Ƶ��
    # ����ǰk������Ƶ����ߵ����
    return sortedClassCount[0][0]


# ���ı���¼��ת��numPy�Ľ�������
def file2matrix(filename):
    # ���ļ����õ��ļ�����
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # ������������
    # �������ص�numPy����
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # �����ļ����ݵ��б�
    for line in arrayOLines:
        line = line.strip()  # ɾ���հ׷�
        listFormLine = line.split('\t')  # splitָ���ָ�����������Ƭ
        returnMat[index, :] = listFormLine[0:3]  # ѡȡǰ3��Ԫ�أ��������洢�ڷ��ؾ�����
        classLabelVector.append(int(listFormLine[-1]))
        # -1������ʾ���һ��Ԫ�أ�Ϊlabel��Ϣ�洢��classLabelVector
        index += 1
    return returnMat, classLabelVector


# ��һ������ֵ
def autoNorm(dataSet):
    minVals = dataSet.min(0);  # ���ÿ����Сֵ������0ʹ�ÿ��Դ�����ѡȡ��Сֵ�������ǵ�ǰ��
    maxVals = dataSet.max(0);  # ���ÿ�����ֵ
    ranges = maxVals - minVals;
    normDataSet = zeros(shape(dataSet))  # ��ʼ����һ������Ϊ��ȡ��dataset
    m = dataSet.shape[0]  # m�����һ��
    # ����������3*1000��min,max��range��1*3��˲���tile���������ݸ��Ƴ��������ͬ��С
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# ���Դ���
def datingClassTest():
    hoRatio = 0.10  # ��������ռ�İٷֱ�
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f " % (errorCount / float(numTestVecs)))

#Ԥ�⣡����
# ����ĳ�˵Ļ�����Ϣ�������ԶԷ�ϲ���̶ȵ�Ԥ��
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierResult - 1])


# 2����дʶ��ϵͳ
# ��һ��32*32�Ķ�����ͼ�����ת����1*1024������
import os, sys


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# ��дʶ����Դ���,������һЩGitHub�ϵĿ�Դ���룬����Щ����û�н��
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # ��ȡĿ¼����
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # �ָ�õ���ǩ  ���ļ��������õ���������
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        hwLabels.append(classStr)  # ����������ǩ
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # �������ݼ�
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %s, the real answer is:%s' % (classifierResult, classStr))
        if (classifierResult != classStr): errorCount += 1.0
    print("\nthe total numbersof errors is : %d" % errorCount)
    print("\nthetotal error rate is: %f" % (errorCount / float(mTest)))

handwritingClassTest()



