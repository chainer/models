import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links import Conv2DActiv

from .general.resize_images import resize_images


class MaskRefine(chainer.Chain):

    def __init__(self):
        super(MaskRefine, self).__init__()
        with self.init_scope():
            self.v0 = chainer.Sequential(
                    Conv2DActiv(64, 16, ksize=3, pad=1),
                    Conv2DActiv(16, 4, ksize=3, pad=1),
            )
            self.v1 = chainer.Sequential(
                    Conv2DActiv(256, 64, ksize=3, pad=1),
                    Conv2DActiv(64, 16, ksize=3, pad=1),
            )
            self.v2 = chainer.Sequential(
                    Conv2DActiv(512, 128, ksize=3, pad=1),
                    Conv2DActiv(128, 32, ksize=3, pad=1),
            )

            self.h2 = chainer.Sequential(
                    Conv2DActiv(32, 32, ksize=3, pad=1),
                    Conv2DActiv(32, 32, ksize=3, pad=1),
            )
            self.h1 = chainer.Sequential(
                    Conv2DActiv(16, 16, ksize=3, pad=1),
                    Conv2DActiv(16, 16, ksize=3, pad=1),
            )
            self.h0 = chainer.Sequential(
                    Conv2DActiv(4, 4, ksize=3, pad=1),
                    Conv2DActiv(4, 4, ksize=3, pad=1),
            )

            self.deconv = L.Deconvolution2D(256, 32, ksize=15, stride=15)
            self.post0 = L.Convolution2D(32, 16, ksize=3, pad=1)
            self.post1 = L.Convolution2D(16, 4, ksize=3, pad=1)
            self.post2 = L.Convolution2D(4, 1, ksize=3, pad=1)

    def forward(self, f, corr_feature, pos):
        p0 = F.pad(f[0], ((0, 0), (0, 0), (16, 16), (16, 16)), 'constant')
        p0 = p0[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]

        p1 = F.pad(f[1], ((0, 0), (0, 0), (8, 8), (8, 8)), 'constant')
        p1 = p1[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]

        p2 = F.pad(f[2], ((0, 0), (0, 0), (4, 4), (4, 4)), 'constant')
        p2 = p2[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]

        p3 = corr_feature[:, :, pos[0], pos[1]].reshape((-1, 256, 1, 1))

        out = self.deconv(p3)
        # NOTE: In the original Torch, resize_images uses 'nearest' interpolation
        out = self.h2(out) + self.v2(p2)
        out = self.post0(resize_images(
            out, (31, 31), align_corners=False, mode='nearest'))
        out = self.h1(out) + self.v1(p1)
        out = self.post1(
                resize_images(out, (61, 61), align_corners=False, mode='nearest'))
        out = self.h0(out) + self.v0(p0)
        out = self.post2(
                resize_images(out, (127, 127), align_corners=False, mode='nearest'))

        return out.reshape((-1, 127 ** 2))
