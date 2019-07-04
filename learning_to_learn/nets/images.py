import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

# The base network is an MLP with one hidden layer of 20 units using a sigmoid activation function


class MLPforMNIST(chainer.Chain):
    def __init__(self, n_units=20, n_classes=10):
        super(MLPforMNIST, self).__init__()
        initializer = chainer.initializers.Normal(scale=0.001)
        with self.init_scope():
            self.l1 = L.Linear(None, n_units, initialW=initializer)
            self.lout = L.Linear(None, n_classes, initialW=initializer)

    def logit(self, x):
        h = self.l1(x)
        h = F.sigmoid(h)
        h = self.lout(h)
        return h

    def __call__(self, x, t, get_accuracy=False):
        logit = self.logit(x)
        loss = F.softmax_cross_entropy(logit, t)
        acc = F.accuracy(logit, t).item()
        if get_accuracy:
            return loss, acc
        else:
            return loss
