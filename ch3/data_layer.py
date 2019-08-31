from algorithm.ann.layer import Layer


class DataLayer(Layer):
    def __init__(self, name, bottomShape, topShape):
        super(DataLayer, self).__init__(
            'DataLayer', name, bottomShape, topShape
        )

    def setUp(self):
        pass

    def forward(self, bottom):
        top = bottom.reshape(self.topShape)

        return top, None

    def backward(self, top, z, bottom, weight, diff):
        pass
