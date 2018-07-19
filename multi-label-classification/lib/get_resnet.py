import numpy as np

import chainer.links as L
from chainercv.links import ResNet50


# Please read the turoial.
# https://chainercv.readthedocs.io/en/stable/tutorial/link.html#fine-tuning
def get_shape_mismatch_names(src, dst):
    # all parameters are assumed to be initialized
    mismatch_names = []
    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        dst_param = dst_named_param[1]
        src_param = src_params[name]
        if src_param.shape != dst_param.shape:
            mismatch_names.append(name)
    return mismatch_names


# https://chainercv.readthedocs.io/en/stable/tutorial/link.html#fine-tuning
# Copy of the example
def get_resnet_50(n_class):
    src = ResNet50(
        n_class=1000, pretrained_model='imagenet', arch='he')
    dst = ResNet50(n_class=n_class, arch='he')
    # initialized weights
    dst(np.zeros((1, 3, 224, 224), dtype=np.float32))

    ignore_names = get_shape_mismatch_names(src, dst)

    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        if name not in ignore_names:
            dst_named_param[1].array[:] = src_params[name].array[:]

    # Copy Batch Normalization's statistics
    dst_links = dict(dst.namedlinks())
    for name, link in src.namedlinks():
        if isinstance(link, L.BatchNormalization):
            dst_bn = dst_links[name]
            dst_bn.avg_mean[:] = link.avg_mean
            dst_bn.avg_var[:] = link.avg_var
    return dst
