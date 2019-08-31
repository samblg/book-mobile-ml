import numpy as np

from algorithm.ann.layers.datalayer import DataLayer
from algorithm.ann.layers.simplelayer import SimpleLayer
from algorithm.ann.net import Net
from common.io.tsv import TsvDataSetReader


def main():
    dataSetReader = TsvDataSetReader()
    dataSet = dataSetReader.loadDataSet('annData.txt',
                                        attrType=float)

    attributes = [item[:-1] for item in dataSet]
    labels = [item[-1] for item in dataSet]

    print(attributes)
    print(labels)

    net = Net(alpha=.5, lamb=.6, maxIteration=10)

    net.addLayers(
        DataLayer('input', bottomShape=(1, 4), topShape=(4, 1)),
        SimpleLayer('hidden1', bottomShape=(4, 1), topShape=(3, 1)),
        SimpleLayer('output', bottomShape=(3, 1), topShape=(1, 1))
    )

    net.setUp()
    net.trainByOne(np.mat(attributes[0]), np.mat(labels[0]))

    print(top)
    print(z)

    pass

if __name__ == '__main__':
    main()
