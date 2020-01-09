import chainer.functions as F
import chainer.links as L

from chainer import Chain

from .utils import SublayerConnection


class Encoder(Chain):
    """
        Base building block for the encoder that holds the number of encoder layers
    """

    def __init__(self, sublayer, N):
        super().__init__()
        with self.init_scope():
            self.sub_layers = sublayer.repeat(N, mode='copy')
            self.norm = L.LayerNormalization(sublayer.size)

    def __call__(self, x, mask):
        """
            The forward pass of the encoder
            :param x: the embedded input data
            :param mask: the mask to apply to the input data during attention calculation
            :return: the output of the encoder
        """
        for sub_layer in self.sub_layers:
            x = sub_layer(x, mask)

        batch_size, num_steps, size = x.shape
        normed_x = self.norm(F.reshape(x, (-1, size)))
        return F.reshape(normed_x, (batch_size, num_steps, size))


class EncoderLayer(Chain):
    """
        Implementation of a single encoder layer.
    """

    def __init__(self, size, self_attention, feed_forward, dropout_ratio=0.1):
        """
            Initialize the encoder.
            :param size: the number of hidden units.
            :param self_attention: the attention layer to use for self attention
            :param feed_forward: the PositionwiseFeedForward layer to use
            :param dropout_ratio: the dropout ratio
        """
        super().__init__()
        self.size = size
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.self_attention = SublayerConnection(self_attention, self.size, self.dropout_ratio)
            self.feed_forward = SublayerConnection(feed_forward, self.size, self.dropout_ratio)

    def __call__(self, x, mask):
        """
            Perform masked self attention, and end with the
            PositionwiseFeedForward layer.
            :param x: the embedded input data
            :param mask: mask used to guide the self attention
            :return: the output of this encoder stage
        """
        x = self.self_attention(x, lambda x: self.self_attention.layer(x, x, x, mask))
        return self.feed_forward(x, self.feed_forward.layer)
