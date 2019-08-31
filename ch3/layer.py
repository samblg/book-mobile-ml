class Layer(object):
    def __init__(self, layerType, name, bottomShape, topShape):
        super(Layer, self).__init__()

        self.type = layerType
        self.name = name
        self.bottomShape = bottomShape
        self.topShape = topShape

    def setUp(self):
        raise NotImplementedError

    def baseForward(self, bottom):
        if bottom.shape != self.bottomShape:
            print('Bottom shape of layer {name} should be: {shape}').format(name=self.name, shape=self.bottomShape)
        assert bottom.shape == self.bottomShape;

        top, z = self.forward(bottom)

        print(top)

    def forward(self, bottom):
        raise NotImplementedError

    def baseBackward(self, top, z, bottom, weight, diff):
        return self.backward(top, z, bottom, weight, diff)

    def backward(self, top, z, bottom, weight, diff):
        raise NotImplementedErrorayer.py
