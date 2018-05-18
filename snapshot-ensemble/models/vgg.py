"""
    VGG model definition
    ported from https://github.com/chainer/chainer/blob/master/examples/cifar/models/VGG.py

"""
import math
import chainer
import chainer.functions as F
import chainer.links as L


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            initializer = chainer.initializers.HeNormal()
            self.conv = L.Convolution2D(in_channels, out_channels, ksize, pad=pad,
                                        initialW=initializer)

    def __call__(self, x):
        h = self.conv(x)
        return F.relu(h)


class VGG16(chainer.Chain):
    def __init__(self, class_labels=10):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(3, 64)
            self.block1_2 = Block(64, 64)
            self.block2_1 = Block(64, 128)
            self.block2_2 = Block(128, 128)
            self.block3_1 = Block(128, 256)
            self.block3_2 = Block(256, 256)
            self.block3_3 = Block(256, 256)
            self.block4_1 = Block(256, 512)
            self.block4_2 = Block(512, 512)
            self.block4_3 = Block(512, 512)
            self.block5_1 = Block(512, 512)
            self.block5_2 = Block(512, 512)
            self.block5_3 = Block(512, 512)

            init = chainer.initializers.Uniform(1. / math.sqrt(512))
            self.fc1 = L.Linear(512, 512, initialW=init, initial_bias=init)
            self.fc2 = L.Linear(512, 512, initialW=init, initial_bias=init)
            self.fc3 = L.Linear(512, class_labels, initialW=init, initial_bias=init)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = self.block3_2(h)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = self.block4_2(h)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = self.block5_2(h)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # classifier
        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc2(h)
        h = F.relu(h)
        return self.fc3(h)
