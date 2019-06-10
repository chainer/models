import chainer
from .dilated_resnet import DilatedResNet
from .neck import AdjustAllLayer
from .rpn import MultiRPN
from .rpn import DepthwiseRPN
from .rpn import DepthwiseXCorr
from .mask_refine import MaskRefine


class SiamRPN(chainer.Chain):

    def __init__(self, multi_scale=True, mask=False):
        super(SiamRPN, self).__init__()

        n_layer = 50
        backbone_channels = [512, 1024, 2048]
        adjust_channels = [256, 256, 256]

        with self.init_scope():
            self.extractor = DilatedResNet(n_layer)
            if multi_scale:
                self.neck = AdjustAllLayer(
                    backbone_channels, adjust_channels)
                self.rpn = MultiRPN(adjust_channels, 5)
            else:
                self.neck = AdjustAllLayer([1024], [256])
                self.rpn = DepthwiseRPN(
                        in_channels=256, out_channels=256, n_anchor=5)

            if mask:
                self.extractor.pick = ('conv1', 'res2', 'res3', 'res4')
                self.mask_head = DepthwiseXCorr(
                        in_channels=256,
                        mid_channels=256,
                        out_channels=63 **2)
                self.mask_refine = MaskRefine()
            else:
                self.extractor.pick = ('res3', 'res4', 'res5')

        self.zs = None
        self.mask = mask

    def template(self, z):
        with chainer.using_config(
            'train', False), chainer.no_backprop_mode():
            zs = self.extractor(z)
            if self.mask:
                zs = zs[-1]

            if hasattr(self, 'neck'):
                zs = self.neck(zs)
            # TODO: Decide whether to keep zs as Variable or ndarray
            self.zs = zs

    def track(self, x):
        with chainer.using_config(
            'train', False), chainer.no_backprop_mode():
            xs = self.extractor(x)

            if self.mask:
                self.xs = xs[:-1]  # TODO
                xs = xs[-1]

            if hasattr(self, 'neck'):
                xs = self.neck(xs)

            conf, loc = self.rpn(self.zs, xs)

            if self.mask:
                mask, h_mask_corr = self.mask_head(self.zs, xs)
                self.h_mask_corr = h_mask_corr

        if self.mask:
            return conf, loc, mask
        else:
            return conf, loc

    def refine_mask(self, pos):
        return self.mask_refine(self.xs, self.h_mask_corr, pos)
