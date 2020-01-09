import chainer.functions as F
import chainer.links as L

from chainer import Chain, initializers


class PositionwiseFeedForward(Chain):
    """
        Implementation of the PositionwiseFeedForward layer in a transformer
    """

    def __init__(self, size, ff_size=2048, dropout_ratio=0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.l1 = L.Linear(size, ff_size, initialW=initializers.GlorotUniform())
            self.l2 = L.Linear(ff_size, size, initialW=initializers.GlorotUniform())

    def __call__(self, x):
        h = F.relu(self.l1(x, n_batch_axes=2))
        h = F.dropout(h, self.dropout_ratio)
        return self.l2(h, n_batch_axes=2)
