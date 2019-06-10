import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links import Conv2DBNActiv
import numpy as np


def xcorr_depthwise(kernel, x):
    """depthwise cross correlation
    """
    B, C, ker_H, ker_W = kernel.shape
    _, _, H, W = x.shape
    x = x.reshape((1, B * C, H, W))
    kernel = kernel.reshape((B * C, 1, ker_H, ker_W))
    out = F.convolution_2d(x, kernel, groups=B * C)
    out_H, out_W = out.shape[2:]
    out = out.reshape((B, C, out_H, out_W))
    return out


class DepthwiseXCorr(chainer.Chain):

    def __init__(
        self, in_channels, mid_channels, out_channels,
        ksize=3, hidden_ksize=5):
        bn_kwargs = {'eps': 1e-5}
        super(DepthwiseXCorr, self).__init__()
        with self.init_scope():
            self.conv_kernel = Conv2DBNActiv(
                in_channels, mid_channels, ksize=ksize, bn_kwargs=bn_kwargs)
            self.conv_search = Conv2DBNActiv(
                in_channels, mid_channels, ksize=ksize, bn_kwargs=bn_kwargs)
            self.conv_head1 = Conv2DBNActiv(
                mid_channels, mid_channels, ksize=1, bn_kwargs=bn_kwargs)
            self.conv_head2 = L.Convolution2D(
                mid_channels, out_channels, ksize=1, nobias=False)

    def forward(self, z, x):
        h_z = self.conv_kernel(z)
        h_x = self.conv_search(x)
        h1 = xcorr_depthwise(h_z, h_x)
        out = self.conv_head2(self.conv_head1(h1))
        return out, h1


class DepthwiseRPN(chainer.Chain):

    def __init__(self, in_channels=256, out_channels=256, n_anchor=5):
        super(DepthwiseRPN, self).__init__()
        with self.init_scope():
            self.conf = DepthwiseXCorr(
                in_channels, out_channels, 2 * n_anchor)
            self.loc = DepthwiseXCorr(
                in_channels, out_channels, 4 * n_anchor)

    def forward(self, z, x):
        conf, _ = self.conf(z, x)
        loc, _ = self.loc(z, x)
        return conf, loc


class MultiRPN(chainer.Chain):

    def __init__(self, in_channels_list, n_anchor):
        super(MultiRPN, self).__init__()

        n_scale = len(in_channels_list)
        with self.init_scope():
            self.conf_weight = chainer.Parameter(
                np.ones((n_scale,), dtype=np.float32))
            self.loc_weight = chainer.Parameter(
                np.ones((n_scale,), dtype=np.float32))

            for i in range(n_scale):
                setattr(
                    self, 'rpn' + str(i + 3),
                    DepthwiseRPN(in_channels_list[i], in_channels_list[i], n_anchor=n_anchor)
                )

        
    def forward(self, zs, xs):
        confs = []
        locs = []
        for i, (z, x) in enumerate(zip(zs, xs)):
            conf, loc = getattr(self, 'rpn' + str(i + 3))(z, x)
            confs.append(conf)
            locs.append(loc)

        conf_weight = F.softmax(self.conf_weight, axis=0)
        loc_weight = F.softmax(self.loc_weight, axis=0)

        return (
            self.weight_average(conf_weight, confs),
            self.weight_average(loc_weight, locs))

    def weight_average(self, weight, value):
        assert len(weight) == len(value)
        acc = 0
        for i in range(len(weight)):
            acc += weight[i] * value[i]
        return acc

