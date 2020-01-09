import math

import chainer.functions as F
import chainer.links as L

from chainer import Chain, initializers


class MultiHeadedAttention(Chain):
    """
        Attention implementation used by the Transformer. The attention implementation uses multiple attention heads.
    """

    def __init__(self, num_heads, size, dropout_ratio=0.1):
        super().__init__()
        assert size % num_heads == 0, "model size must be divisible by the number of heads"

        self.key_dimensionality = size // num_heads
        self.num_heads = num_heads
        self.attention = None
        self.dropout_ratio = dropout_ratio

        with self.init_scope():
            self.linears = L.Linear(size, size, initialW=initializers.GlorotUniform()).repeat(4, mode='init')

    def project(self, linear_function, weight_matrix, batch_size):
        weight_matrix = linear_function(weight_matrix, n_batch_axes=2)
        weight_matrix = F.reshape(weight_matrix, (batch_size, -1, self.num_heads, self.key_dimensionality))
        return F.transpose(weight_matrix, (0, 2, 1, 3))

    def attention_implementation(self, query, key, value, mask=None, dropout_ratio=None):
        scores = F.matmul(query, F.transpose(key, (0, 1, 3, 2))) / math.sqrt(self.key_dimensionality)
        if mask is not None:
            batch_size, num_heads, _, _ = scores.shape
            mask = self.xp.array(mask)
            mask = self.xp.broadcast_to(mask, (batch_size, num_heads) + mask.shape[2:])
            mask = mask[:, :, :scores.shape[2], :scores.shape[3]]
            scores = F.where(mask, scores, self.xp.full_like(scores.array, -1e9))

        attention_probabilities = F.softmax(scores, axis=3)
        if dropout_ratio is not None:
            attention_probabilities = F.dropout(attention_probabilities, ratio=dropout_ratio)

        return F.matmul(attention_probabilities, value), attention_probabilities

    def __call__(self, query, key, value, mask=None):
        """
            Perform attention on the value array, using the query and key parameters for calculating the attention mask.
            :param query: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention mask calculation
            :param key: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention mask calculation
            :param value: matrix of shape (batch_size, num_timesteps, transformer_size) that is used for attention calculation
            :param mask: mask that can be used to mask out parts of the feature maps and avoid attending to those parts
            :return: the attended feature map `value`.
        """
        if mask is not None:
            mask = mask[:, self.xp.newaxis, ...]

        batch_size = len(query)

        query, key, value = [self.project(linear, x, batch_size) for linear, x in zip(self.linears, (query, key, value))]

        x, self.attention = self.attention_implementation(query, key, value, mask=mask, dropout_ratio=self.dropout_ratio)

        x = F.transpose(x, (0, 2, 1, 3))
        x = F.reshape(x, (batch_size, -1, self.num_heads * self.key_dimensionality))

        return self.linears[-1](x, n_batch_axes=2)
