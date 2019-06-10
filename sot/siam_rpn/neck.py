import chainer
import chainer.functions as F
from chainercv.links import Conv2DBNActiv


class AdjustLayer(chainer.Chain):

    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        with self.init_scope():
            self.conv = Conv2DBNActiv(
                in_channels, out_channels, ksize=1, bn_kwargs={'eps': 1e-5},
                activ=None
                )

    def forward(self, x):
        h = self.conv(x)
        _, _, H, W = h.shape
        if H < 20:
            y_slice = slice(4, 11)
        else:
            y_slice = slice(0, H)
        if W < 20:
            x_slice = slice(4, 11)
        else:
            x_slice = slice(0, W)
        h = h[:, :, y_slice, x_slice]
        return h


class AdjustAllLayer(chainer.Chain):

    def __init__(self, in_channels_list, out_channels_list):
        super(AdjustAllLayer, self).__init__()
        self.n_scale = len(out_channels_list)
        with self.init_scope():
            if self.n_scale == 1:
                self.downsample = AdjustLayer(
                    in_channels_list[0], out_channels_list[0])
            else:
                for i in range(self.n_scale):
                    setattr(
                        self, 'downsample' + str(i + 3),
                        AdjustLayer(in_channels_list[i], out_channels_list[i]))

    def forward(self, xs):
        if self.n_scale == 1:
            return self.downsample(xs)
        else:
            outs = []
            for i in range(self.n_scale):
                l = getattr(self, 'downsample' + str(i + 3))
                outs.append(l(xs[i]))
            return outs
