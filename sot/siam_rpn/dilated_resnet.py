import chainer.functions as F

from chainercv.links import Conv2DBNActiv
from chainercv.links import PickableSequentialChain
from chainercv import transforms
from chainercv import utils

from .resblock import ResBlock


class DilatedResBlock(ResBlock):

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, dilate=1, groups=1, initialW=None,
                 bn_kwargs={}, stride_first=False,
                 add_seblock=False):
        if stride == 1 and dilate == 1:
            residual_conv = None
        else:
            ksize = 3
            if dilate > 1:
                dd = dilate // 2
                pad = dd
            else:
                dd = 1
                pad = 0
            residual_conv = Conv2DBNActiv(
                in_channels, out_channels, ksize, stride,
                pad, dilate=dd,
                nobias=True, initialW=initialW,
                activ=None, bn_kwargs=bn_kwargs)
        super(DilatedResBlock, self).__init__(
            n_layer, in_channels, mid_channels,
            out_channels, stride, dilate, groups, initialW,
            bn_kwargs, stride_first,
            residual_conv, add_seblock
        )


class DilatedResNet(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    def __init__(self, n_layer, pretrained_model=None,
                 initialW=None):
        n_block = self._blocks[n_layer]

        # _, path = utils.prepare_pretrained_model(
        #     {},
        #     pretrained_model,
        #     self._models[n_layer])

        bn_kwargs = {'eps': 1e-5}

        super(DilatedResNet, self).__init__()
        with self.init_scope():
            # pad is not 3
            self.conv1 = Conv2DBNActiv(
                3, 64, 7, 2, 0, initialW=initialW,
                bn_kwargs=bn_kwargs)
            # pad is 1
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1)
            self.res2 = DilatedResBlock(
                n_block[0], 64, 64, 256, 1, 1,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )
            self.res3 = DilatedResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )

            # 
            self.res4 = DilatedResBlock(
                n_block[2], 512, 256, 1024, 1, 2,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs,
                )
            self.res5 = DilatedResBlock(
                n_block[3], 1024, 512, 2048, 1, 4,
                initialW=initialW, stride_first=False,
                bn_kwargs=bn_kwargs
                )

        # if path:
        #     chainer.serializers.load_npz(path, self, ignore_names=None)
