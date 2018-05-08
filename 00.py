# -*- coding: cp936 -*-

from numpy import *
import operator
from os import listdir


# 创建数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 第一个kNN分类器  inX-测试数据 dataSet-样本数据  labels-标签 k-邻近的k个样本
def classify0(inX, dataSet, labels, k):
    # 数据集大小
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 是幂运算，这里用的是欧氏距离
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 按距离排序
    sortedDistIndicies = distances.argsort()  # 返回数值从小到大的索引值
    # 统计前k个点所属的类别
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        # 根据排序结果的索引值返回靠近的前k个标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 各个标签出现频率
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    # sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序频率
    # 返回前k个点钟频率最高的类别
    return sortedClassCount[0][0]


# 将文本记录到转换numPy的解析程序
def file2matrix(filename):
    # 打开文件并得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 读出数据行数
    # 创建返回的numPy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()  # 删除空白符
        listFormLine = line.split('\t')  # split指定分隔符对数据切片
        returnMat[index, :] = listFormLine[0:3]  # 选取前3个元素（特征）存储在返回矩阵中
        classLabelVector.append(int(listFormLine[-1]))
        # -1索引表示最后一列元素，为label信息存储在classLabelVector
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0);  # 存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0);  # 存放每列最大值
    ranges = maxVals - minVals;
    normDataSet = zeros(shape(dataSet))  # 初始化归一化矩阵为读取的dataset
    m = dataSet.shape[0]  # m保存第一行
    # 特征矩阵是3*1000，min,max，range是1*3因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试代码
def datingClassTest():
    hoRatio = 0.10  # 测试数据占的百分比
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

#预测！！！
# 输入某人的基本信息，做出对对方喜欢程度的预测
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


# 2：手写识别系统
# 将一个32*32的二进制图像矩阵转换成1*1024的向量
import os, sys


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写识别测试代码,参照了一些GitHub上的开源代码，还有些问题没有解决
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 分割得到标签  从文件名解析得到分类数据
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        hwLabels.append(classStr)  # 测试样例标签
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 测试数据集
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



