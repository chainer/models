# -*- coding: utf-8 -*-

import random

import chainer
import numpy as np
import six
from chainer.serializers import DictionarySerializer


def set_random_seed(seed, gpu):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)
    # set CuPy random seed
    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        chainer.backends.cuda.cupy.random.seed(seed)


def save_non_embed_npz(file, obj, compression=True):
    """
    Since we do not update the original embedding matrix,
    it is unnecessary (and waste of storage) to include it in model file.
    --> Exclude embed_mat from .npz file
    """
    if isinstance(file, six.string_types):
        with open(file, 'wb') as f:
            save_non_embed_npz(f, obj, compression)
        return

    s = DictionarySerializer()
    s.save(obj)
    s.target = {k: v for k, v in s.target.items() if 'embed_mat' not in k}
    if compression:
        np.savez_compressed(file, **s.target)
    else:
        np.savez(file, **s.target)
