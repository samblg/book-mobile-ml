from common.tree import Tree, TreeNode
from common.math import expectation
import math

class DecisionTree(Tree):
    def __init__(self, **kwargs):
        # 调用父类初始化方法
        # 新式类鼓励使用这种方式初始化
        super(DecisionTree, self).__init__(**kwargs)

    def train(self, dataSet, featureNames):
        """使用训练集训练，featureNames指定训练集中每个特征的名称"""
        if len(dataSet) == 0:
            self.root = None
            return

        # 调用createTreeNode创建决策树根节点
        self.root = DecisionTree.createTreeNode(dataSet, featureNames)

    @staticmethod
    def createTreeNode(dataSet, featureNames):
        """创建决策树节点"""
        # 获取当前数据集中的所有标签
        labels = [item[-1] for item in dataSet]
        # 过滤重复标签
        uniqueLabels = set(labels)
        # 如果当前数据集中标签都一样返回叶子节点
        if len(uniqueLabels) == 1:
            return TreeNode(label=labels[0])

        # 如果当前数据集中所有数据特征相同，返回叶子节点
        if DecisionTree.isAllItemIdentical(dataSet):
            # 叶子节点标签取主要的标签（出现次数最多的标签）
            majorLabel = DecisionTree.majorLabel(labels)
            return TreeNode(label=majorLabel)

        # 选择一个最佳的划分特征
        bestFeature = DecisionTree.chooseBestSplitFeature(dataSet)
        bestFeatureName = featureNames[bestFeature]
        featureNames.pop(bestFeature)
        children = {}
        # 获取划分特征的所有可能值
        featureValues = set([item[bestFeature] for item in dataSet])

        # 遍历特征的所有值
        for value in featureValues:
            # 使用该值划分子数据集
            subDataSet = DecisionTree.splitDataSet(dataSet, bestFeature, value)
            # 继续根据子数据集创建决策树节点
            children[value] = DecisionTree.createTreeNode(subDataSet, featureNames[:])

        # 返回树节点
        return TreeNode(label=bestFeatureName, children=children)

    @staticmethod
    def isAllItemIdentical(dataSet):
        """判断某个数据集中是否所有特征都相同"""
        # 如果没有数据或者只有一个数据那么返回True
        if len(dataSet) == 0 or len(dataSet) == 1:
            return True

        # 获取除了标签外的所有特征
        itemList = [item[:-1] for item in dataSet]
        # 逐个比较数据项，如果元素不等返回False
        prevItem = itemList[0]
        for item in itemList:
            if item != prevItem:
                return False

            prevItem = item

        # 数据项全部相同，返回True
        return True

    @staticmethod
    def majorLabel(labels):
        labelCounts = {}
        for label in labels:
            labelCount = labelCounts.get(label, 0)
            labelCounts[label] = labelCount + 1

        labelCountPairs = list(labelCounts.items())
        labelCountPairs.sort(key=lambda pair: pair[1], reverse=True)

        return labelCountPairs[0][0]

    @staticmethod
    def chooseBestSplitFeature(dataSet):
        itemCount = len(dataSet)
        if itemCount == 0:
            return

        minEntropy = math.inf
        bestFeature = -1
        featureCount = len(dataSet[0]) - 1
        for axis in range(featureCount):
            featureValues = set([item[axis] for item in dataSet])
            subDataSets = [DecisionTree.splitDataSet(dataSet, axis, value) for value in featureValues]
            subEntropies = [shannonEntropy(subDataSet) for subDataSet in subDataSets]
            probabilities = [len(subDataSet) / itemCount for subDataSet in subDataSets]

            newEntropy = expectation(probabilities, subEntropies)
            if newEntropy < minEntropy:
                minEntropy = newEntropy
                bestFeature = axis

        return bestFeature

    @staticmethod
    def splitDataSet(dataSet, splitAxis, value):
        results = [item[:splitAxis] + item[splitAxis + 1:] for item in dataSet if item[splitAxis] == value]
        return results

    # 预测一组数据
    def predict(self, items):
        result = [self.predictOne(item) for item in items]

        return result

    # 预测单个数据
    def predictOne(self, item, **kwargs):
        # 如果是叶子节点，说明已经获得了预测的分类标签
        currentNode = kwargs.get('node', self.root)
        if currentNode.isLeaf():
            return currentNode.label

        # 否则说明是决策节点，需要获取特征名称（节点标签）
        featureName = currentNode.label

        # 如果数据项中没有对应特征则抛出异常
        featureValue = item.get(featureName, None)
        if featureValue is None:
            raise KeyError('''Item don't have the key '%s'.''' % featureName)

        # 如果节点没有对应的分支则抛出异常
        nextNode = currentNode.children.get(featureValue, None)
        if nextNode is None:
            raise KeyError('''TreeNode don't have the branch '%s'.''' % featureValue)

        # 递归预测
        return self.predictOne(item, node=nextNode)


def shannonEntropy(dataSet):
    labelCounts = {}
    for item in dataSet:
        label = item[-1]
        labelCount = labelCounts.get(label, 0)
        labelCounts[label] = labelCount + 1

    itemCount = len(dataSet)
    probabilities = [labelCount / itemCount for labelCount in labelCounts.values()]

    return sum([shannonInfo(probability) for probability in probabilities])


def shannonInfo(probability):
    return -probability * math.log(probability, 2)
