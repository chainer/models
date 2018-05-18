"""
    wide_resnet model definition
    ported from https://github.com/mitmul/chainer-cifar10/blob/master/models/wide_resnet.py
"""
import chainer
import chainer.functions as F
import chainer.links as L


class WideBasic(chainer.Chain):
    def __init__(self, n_input, n_output, stride, dropout_rate=0.0):
        w = chainer.initializers.HeNormal()
        super(WideBasic, self).__init__()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(n_input)
            self.conv1 = L.Convolution2D(
                n_input, n_output, 3, 1, 1, nobias=False, initialW=w)
            self.bn2 = L.BatchNormalization(n_output)
            self.conv2 = L.Convolution2D(
                n_output, n_output, 3, stride, 1, nobias=False, initialW=w)

            if stride != 1 or n_input != n_output:
                self.shortcut = L.Convolution2D(
                    n_input, n_output, 1, stride, nobias=False, initialW=w)
            self.dropout_ratio = dropout_rate

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_ratio != 0:
            h = F.dropout(h, self.dropout_rate)
        h = self.conv2(F.relu(self.bn2(h)))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return h + shortcut


class WideBlock(chainer.ChainList):
    def __init__(self, n_input, n_output, count, stride, dropout):
        super(WideBlock, self).__init__()
        self.add_link(WideBasic(n_input, n_output, stride, dropout))
        for _ in range(count - 1):
            self.add_link(WideBasic(n_output, n_output, 1, dropout))

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class WideResNet28x10(chainer.Chain):
    def __init__(
            self, num_classes=10, widen_factor=10, depth=28, dropout_rate=0):
        k = widen_factor
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        w = chainer.initializers.HeNormal()
        super(WideResNet28x10, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, n_stages[0], 3, 1, 1, nobias=False, initialW=w)
            self.wide2 = WideBlock(n_stages[0], n_stages[1], n, 1, dropout_rate)
            self.wide3 = WideBlock(n_stages[1], n_stages[2], n, 2, dropout_rate)
            self.wide4 = WideBlock(n_stages[2], n_stages[3], n, 2, dropout_rate)
            self.bn5 = L.BatchNormalization(n_stages[3], decay=0.1)
            self.fc6 = L.Linear(n_stages[3], num_classes, initialW=w)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.wide2(h)
        h = self.wide3(h)
        h = self.wide4(h)
        h = F.relu(self.bn5(h))
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        return self.fc6(h)
