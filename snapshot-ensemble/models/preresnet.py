"""
    pre-activation resenet model definition
    ported from: https://github.com/mitmul/chainer-cifar10/blob/master/models/resnet.py
"""
import chainer
import chainer.functions as F
import chainer.links as L


class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, use_conv=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(n_in)
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, 1, 0, True, w)
            self.bn2 = L.BatchNormalization(n_mid)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, stride, 1, True, w)
            self.bn3 = L.BatchNormalization(n_mid)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w)
            if use_conv:
                self.conv4 = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, True, w)
        self.use_conv = use_conv

    def __call__(self, x):
        residual = x

        h = self.bn1(x)
        h = F.relu(h)
        h = self.conv1(h)

        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)

        h = self.bn3(h)
        h = F.relu(h)
        h = self.conv3(h)

        if self.use_conv:
            residual = self.conv4(x)

        h += residual

        return h


class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
        for _ in range(n_bottlenecks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class PreResNet110(chainer.Chain):

    def __init__(self, n_class=10, n_blocks=[18, 18, 18]):
        super(PreResNet110, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 1, 1, True, w)
            self.res3 = Block(16, 16, 64, n_blocks[0], 1)
            self.res4 = Block(64, 32, 128, n_blocks[1], 2)
            self.res5 = Block(128, 64, 256, n_blocks[2], 2)
            self.bn6 = L.BatchNormalization(256)
            self.fc7 = L.Linear(256, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.relu(self.bn6(h))
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc7(h)
        return h
