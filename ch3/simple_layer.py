from algorithm.ann.layer import Layer
import numpy as np


class SimpleLayer(Layer):
    def __init__(self, name, bottomShape, topShape):
        super(SimpleLayer, self).__init__(
            'SimpleLayer', name, bottomShape, topShape
        )

        self.weight = None
        self.bias = None

    def setUp(self):
        # s1
        s1 = self.bottomShape[0]
        # s2
        s2 = self.topShape[0]

        # s2 * s1, random
        self.weight = np.mat(np.random.uniform(-0.5, 0.5, (s2, s1)))
        self.bias = np.mat(np.zeros((s2, 1,), dtype=np.float))

    def forward(self, bottom):
        z = self.weight * bottom + self.bias
        top = SimpleLayer.sigmoid(z)

        return top, z

    def backward(self, top, z, bottom, weight, diff):
        diff = np.multiply(weight.T * diff, SimpleLayer.derivative(z))

        return self.weight, diff

    @staticmethod
    def sigmoid(mat):
        return 1.0/(1 + np.exp(-mat))

    @staticmethod
    def derivative(mat):
        fz = SimpleLayer.sigmoid(mat)

        return np.multiply(fz, (1 -fz))
