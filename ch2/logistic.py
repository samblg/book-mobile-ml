from numpy import *
import pickle

class LogisticClassfier:
    # 训练，暂不实现
    def train():
        pass

    # 预测并返回分类结果
    def predict(x):
        # 根据权重和输入求线性回归值
        y = sum(x * weights)
        # 使用Sigmoid转变为概率
        prob = LogisticClassfier.sigmoid(y)

        # 概率大于0.5分类到1
        if prob > 0.5:
            return 1.0
        
        return 0.0
    
    # 保存模型
    def dump(modelFileName):
        params = dict()
        params['weights'] = self.weights
        
        model = {
            'params': params,
        }
        modelFile = open(modelFileName, 'wb')
        pickle.dump(model, modelFile, 2)
        
        modelFile.close()

    # 加载模型
    def load(modelFileName):
        modelFile = open(modelFileName, 'rb')
        model = pickle.load(modelFile)

        params = model['params']
        self.weights = params['weights']

        modelFile.close()

    # Sigmoid函数实现
    @staticmethod
    def sigmoid(inX):
        return 1.0/(1+exp(-inX))

if __name__ == '__main__':
    # 构建分类器
    classfier = LogisticClassfier()
    # 加载模型
    classfier.load('horse.model')

    # 读取测试集
    frTest=open('data/horseColicTest.txt')

    # 初始化测试结果
    errorCount = 0
    numTestVector = 0.0

    # 每次读取一行
    for line in frTest.readlines():
        numTestVector += 1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        
        # 前21个数据为特征，读取特征
        for i in range(21):
            lineArr.append(float(currLine[i]))

        # 获取预测结果
        predictResult = classfier.predict(array(lineArr))
        # 第22个数据为标签
        isHorseLabel = currLine[21]

        # 判断是否预测正确
        if int(predictResult)!=int(isHorseLabel):
            errorCount += 1

    # 计算输出错误率
    errorRate = errorCount / numTestVector
    print('错误率：%f' % (float(errorRate)))
