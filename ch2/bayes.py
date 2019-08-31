#encoding:UTF-8

from numpy import *
import pickle
import time
        
CLASS_SPAM        = 1
CLASS_NOT_SPAM    = 0

class NaiveBayesMailClassfier:
    def train(mailList, mailClasses):
        # 构建所有邮件包含的单词集合
        self.listVocab = NaiveBayesMailClassfier.createVocabList(mailList)
        
        docNum = len(mailList)
        testSetNum  = int(docNum * 0.1);
        
        # 建立与所有文档等长的空数据集（索引）
        trainingIndexSet = range(docNum)
        # 空测试集
        testSet = []
        
        # 随机索引，用作测试集, 同时将随机的索引从训练集中剔除
        for i in range(testSetNum):
            randIndex = int(random.uniform(0, len(trainingIndexSet)))
            testSet.append(trainingIndexSet[randIndex])
            del(trainingIndexSet[randIndex])
        
        trainMatrix = []
        trainClasses = []
       
        # 生成训练数据
        for docIndex in trainingIndexSet:
            trainMatrix.append(NaiveBayesMailClassfier.bagOfWords2VecMN(listVocab, mailList[docIndex]))
            trainClasses.append(mailClasses[docIndex])
        
        # 训练
        trainNaiveBayes(array(trainMatrix), array(trainClasses))
        
    # 训练朴素贝叶斯模型
    def trainNaiveBayes(trainMatrix, trainClasses):
        numTrainDocs = len(trainMatrix)
        #计算矩阵列数, 等于每个向量的维数
        numWords = len(trainMatrix[0]) 
        numIsSpam  = len(filter(lambda x: x == CLASS_SPAM, trainClasses))
        self.pClassSpam = numIsSpam / float(numTrainDocs)
        pSpamNum = ones(numWords)
        pNotSpamNum = ones(numWords)
        pSpamDenom = 2.0
        pNotSpamDenom = 2.0
        
        for i in range(numTrainDocs):
            if trainClasses[i] == CLASS_SPAM:
                pSpamNum += trainMatrix[i]
                pSpamDenom += sum(trainMatrix[i])
            else:
                pNotSpamNum += trainMatrix[i]
                pNotSpamDenom += sum(trainMatrix[i])
            
        self.pSpamVector = log(pSpamNum / pSpamDenom)
        self.pNormalVector = log(pNotSpamNum / pNotSpamDenom)
        
    def predict(self):
        text = NaiveBayesMailClassfier.parseText(text)
        vecWord = NaiveBayesMailClassfier.bagOfWords2VecMN(listVocab, text)
        classType = self.classifyNaiveBayes(array(vecWord))
            
        return classType

    # 根据邮件单词向量分类，返回垃圾或者非垃圾邮件
    def classifyNaiveBayes(vec2Classify):
        pIsSpam = sum(vec2Classify * self.pSpamVector) + log(self.pClassSpam)    #element-wise mult
        pIsNormal = sum(vec2Classify * self.pNormalVector) + log(1.0 - self.pClassSpam)
        
        if pIsSpam > pIsNormal:
            return CLASS_SPAM
        else: 
            return CLASS_NOT_SPAM
    
    def dump(self, modelFileName):
        params = dict()
        params['pSpamVector'] = self.pSpamVector
        params['pNormalVector'] = self.pNormalVector
        params['pClassSpam'] = self.pClassSpam
        
        model = {
            'params': params,
            'vocab': self.listVocab
        }
        modelFile = open(modelFileName, 'wb')
        pickle.dump(model, modelFile, 2)
        modelFile.close()

    def load(self, modelFileName):
        modelFile = open(modelFileName, 'rb')
        model = pickle.load(modelFile)

        params = model['params']
        self.pSpamVector = params['pSpamVector']
        self.pNormalVector = params['pNormalVector']
        self.pClassSpam = params['pClassSpam']
        self.listVocab = model['vocab']
        modelFile.close()

    # 根据文档中出现的所有单词创建词集
    @staticmethod
    def createVocabList(documentSet):
        vocabSet = set([])
        for document in documentSet:
            vocabSet = vocabSet | set(document)
        return list(vocabSet)
        
    # 从文件中读取已经分词好的训练集
    @staticmethod
    def readWords(fileName):
        wordsFile = open(fileName)
        fileContent = wordsFile.read()
        listOfWord = fileContent.split('\t')
        listOfWord = [x for x in listOfWord if x != ' ']

        wordsFile.close()

        return listOfWord

    # 解析文本，主要是分词并返回分词结果
    @staticmethod
    def parseText(text):
        words = segmentWords(text)
        listOfWord  = []
        for word in words:
            listOfWord.append(word)
        return listOfWord

    # 将输入转化为向量，其所在空间维度为 len(listVocab)
    @staticmethod
    def bagOfWords2VecMN(listVocab, inputSet):
        returnVec = [0]*len(listVocab)
        for word in inputSet:
            if word in listVocab:
                returnVec[listVocab.index(word)] += 1
        return returnVec
        
    
if __name__ == '__main__':
    classfier = NaiveBayesMailClassfier()

    mailList = []
    mailClasses = []
    
    # 读取1000封垃圾邮件
    for i in range(0, 1000):
        wordList = NaiveBayesMailClassfier.readWords('spam/%d.txt' % i)
        mailList.append(wordList)
        mailClasses.append(CLASS_SPAM)

    # 读取1000封非垃圾邮件
    for i in range(0, 1000):
        wordList = NaiveBayesMailClassfier.readWords('mail/%d.txt' % i)
        mailList.append(wordList)
        mailClasses.append(CLASS_NOT_SPAM)

    # 训练并保存模型
    classfier.train(mailList, mailClasses)
    classfier.dump('spam.model')

    # 读取模型并预测
    classfier.load('spam.model')
    text = raw_input('input the text:')
    classType = classfier.predict(text)
    print('Result :%d' % classType)
