# from typing import List

import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

ConfigurationError = ValueError


class ScalarMix(chainer.Chain):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(self, mixture_size, do_layer_norm=False):
        super(ScalarMix, self).__init__()

        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        with self.init_scope():
            self.scalar_parameters = chainer.Parameter(
                0., shape=(mixture_size, ))
            self.gamma = chainer.Parameter(1., shape=(1, ))

    def forward(self, tensors,
                mask=None):
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError("{} tensors were passed, but the module was initialized to "
                                     "mix {} tensors.".format(len(tensors), self.mixture_size))

        # TODO: check why using mask and arranged layernorm
        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = F.sum(tensor_masked) / num_elements_not_masked
            variance = F.sum(
                ((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / F.sqrt(variance + 1E-12)

        normed_weights = F.softmax(self.scalar_parameters[None], axis=1)[0]
        normed_weights = F.split_axis(normed_weights, normed_weights.shape[0],
                                      axis=0)
        # TODO: remove for-loop by broadcast

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                weight = F.broadcast_to(weight[None, None], tensor.shape)
                pieces.append(weight * tensor)
            gamma = F.broadcast_to(self.gamma[None, None], tensor.shape)
            return gamma * sum(pieces)

        else:
            mask_float = mask
            broadcast_mask = F.expand_dims(mask_float, -1)
            input_dim = tensors[0].shape(-1)
            num_elements_not_masked = F.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                weight = F.broadcast_to(weight[None, None], tensor.shape)
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            gamma = F.broadcast_to(self.gamma[None, None], tensor.shape)
            return gamma * sum(pieces)
