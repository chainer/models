import chainer
import chainer.functions as F
import chainer.links as L


class Convnet(chainer.Chain):

    def __init__(self, n_out):
        super(Convnet, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Convolution2D(None, 32, 3, 1, 1),
            l2=L.Convolution2D(None, 32, 3, 1, 1),
            l3=L.Convolution2D(None, 32, 3, 1, 1),
            fc=L.Linear(None, n_out),
        )

    def __call__(self, x):
        B = x.shape[0]
        h1 = F.relu(self.l1(x))  # 28
        h2 = F.relu(self.l2(h1))
        h2 = F.max_pooling_2d(h2, ksize=4, stride=2, pad=1)  # 14
        h3 = F.relu(self.l3(h2))
        self.feat = h3
        return self.fc(h3.reshape(B, -1))


class DeformableConvnet(chainer.Chain):

    def __init__(self, n_out):
        super(DeformableConvnet, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Convolution2D(None, 32, 3, 1, 1),
            l2=L.DeformableConvolution2D(None, 32, 3, 1, 1),
            l3=L.DeformableConvolution2D(None, 32, 3, 1, 1),
            fc=L.Linear(None, n_out),
        )

    def __call__(self, x):
        B = x.shape[0]
        h1 = F.relu(self.l1(x))  # 28
        h2 = F.relu(self.l2(h1))
        h2 = F.max_pooling_2d(h2, ksize=4, stride=2, pad=1)  # 14
        h3 = F.relu(self.l3(h2))
        self.feat = h3
        return self.fc(h3.reshape(B, -1))
