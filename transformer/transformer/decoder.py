import chainer.functions as F
import chainer.links as L

from chainer import Chain

from .utils import SublayerConnection


class Decoder(Chain):
    """
        Base building block for the decoder that holds the number of decoder layers
    """

    def __init__(self, sublayer, N):
        super().__init__()
        with self.init_scope():
            self.sub_layers = sublayer.repeat(N, mode='copy')
            self.norm = L.LayerNormalization(sublayer.size)

    def __call__(self, x, memory, src_mask, tgt_mask):
        """
            forward pass of the decoder that passes the data through all decoder layers
            :param x: the embedded input data
            :param memory: the output of the encoder
            :param src_mask: mask used to guide the attention of the encoder output
            :param tgt_mask: mask used to guide the self attention of the decoder
            :return: the decoded output
        """
        for layer in self.sub_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        batch_size, num_steps, size = x.shape
        normed_x = self.norm(F.reshape(x, (-1, size)))
        return F.reshape(normed_x, (batch_size, num_steps, size))


class DecoderLayer(Chain):
    """
        Implementation of a single decoder layer.
    """

    def __init__(self, size, self_attention, src_attention, feed_forward, dropout_ratio=0.1):
        """
            Initialize the decoder.
            :param size: the number of hidden units.
            :param self_attention: the attention layer to use for self attention
            :param src_attention: the attention layer to use for attending to the output of the encoder
            :param feed_forward: the PositionwiseFeedForward layer to use
            :param dropout_ratio: the dropout ratio
        """
        super().__init__()
        self.size = size
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.self_attention = SublayerConnection(self_attention, self.size, self.dropout_ratio)
            self.src_attention = SublayerConnection(src_attention, self.size, self.dropout_ratio)
            self.feed_forward = SublayerConnection(feed_forward, self.size, self.dropout_ratio)

    def __call__(self, x, memory, src_mask, tgt_mask):
        """
            Perform masked self attention and attention to the encoded feature map, and end with the
            PositionwiseFeedForward layer.
            :param x: the embedded input data
            :param memory: the output of the encoder
            :param src_mask: mask used to guide the attention of the encoder output
            :param tgt_mask: mask used to guide the self attention of the decoder
            :return: the output of this decoder stage
        """
        x = self.self_attention(x, lambda x: self.self_attention.layer(x, x, x, tgt_mask))
        x = self.src_attention(x, lambda x: self.src_attention.layer(x, memory, memory, src_mask))
        return self.feed_forward(x, lambda x: self.feed_forward.layer(x))


