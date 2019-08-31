import numpy as np


class Net(object):
    def __init__(self, alpha, lamb, maxIteration):
        super(Net, self).__init__()

        self.alpha = alpha
        self.lamb = lamb
        self.maxIteration = maxIteration
        self.layers = []
        self.blobs = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def addLayers(self, *args):
        for layer in args:
            self.addLayer(layer)

    def setUp(self):
        for layer in self.layers:
            layer.setUp()

    def forward(self, inputData):
        self.blobs = []
        top = inputData
        z = None

        for layer in self.layers:
            top, z = layer.forward(top)
            self.blobs.append((top, z))

        return top, z

    def train(self, trainItems, labelItems):
        for iteration in range(0, self.maxIteration):
            for trainItem, labelItem in zip(trainItems, labelItems):
                self.trainByOne(trainItem, labelItem)
            # get sum of weight and bias
            # adjust weight of layers

    def trainByOne(self, trainItem, labelItem):
        for layer in self.layers:
            layer.setUp()

        predictResult = self.forward(trainItem)

        y = labelItem
        a = predictResult[0]
        z = predictResult[1]

        diff = -(y - a)
        weightWidth = diff.shape[0]
        weight = np.ones((weightWidth, weightWidth), dtype=np.float)
        # 8 * 8
        weight /= weightWidth

        diffs = []
        for reverseIndex, layer in enumerate(self.layers[-1:0:-1]):
            layerIndex = -(reverseIndex + 1)
            prevLayerIndex = layerIndex - 1

            top, z = self.blobs[layerIndex]
            bottom = self.blobs[prevLayerIndex][0]

            weight, diff = layer.baseBackward(top, z, bottom, weight, diff)
            diffs.insert(0, diff)

        weightDelta = []
        biasDelta = []

        for reverseIndex, layer in enumerate(self.layers[-1:0:-1]):
            layerIndex = -(reverseIndex + 1)
            top = self.blobs[layerIndex - 1][0]
            diff = diffs[layerIndex]

            partDerivate = diff * top.T
            weightDelta.insert(0, partDerivate)
            biasDelta.insert(0, diff)

        print(weightDelta)
        print(biasDelta)
