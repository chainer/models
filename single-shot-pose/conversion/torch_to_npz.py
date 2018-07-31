import argparse

from darknet_torch import Darknet as TorchDarknet

import chainer
from chainercv.links import Conv2DBNActiv
import chainer.functions as F
import chainer.links as L
import numpy as np

import sys
sys.path.append('..')
from lib.ssp import SSPYOLOv2


def copy_conv_bn_activ(src, dst, copy_bn=True):
    weight = src[0].weight.data.numpy()
    dst.conv.W.data[:] = weight

    if copy_bn:
        bn_bias = src[1].bias.data.numpy()
        bn_weight = src[1].weight.data.numpy()
        bn_avg_var = src[1].running_var.numpy()
        bn_avg_mean = src[1].running_mean.numpy()

        dst.bn.avg_mean.data[:] = bn_avg_mean
        dst.bn.avg_var.data[:] = bn_avg_var
        dst.bn.gamma.data[:] = bn_weight
        dst.bn.beta.data[:] = bn_bias


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/yolo-pose.cfg')
    parser.add_argument('--skip-last', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    args = parser.parse_args()
    # Load torch model
    cfgfile = args.cfg
    weightfile = args.weight
    model = TorchDarknet(cfgfile)
    model.print_network()
    if args.skip_last:
        model.load_weights_until_last(weightfile)
    else:
        model.load_weights(weightfile)

    chainer_model = SSPYOLOv2()
    d = {0: chainer_model.conv1,
         2: chainer_model.conv2,
         4: chainer_model.conv3,
         5: chainer_model.conv4,
         6: chainer_model.conv5,
         8: chainer_model.conv6,
         9: chainer_model.conv7,
         10: chainer_model.conv8,
         12: chainer_model.conv9,
         13: chainer_model.conv10,
         14: chainer_model.conv11,
         15: chainer_model.conv12,
         16: chainer_model.conv13,
         18: chainer_model.conv14,
         19: chainer_model.conv15,
         20: chainer_model.conv16,
         21: chainer_model.conv17,
         22: chainer_model.conv18,
         23: chainer_model.conv19,
         24: chainer_model.conv20,
         26: chainer_model.conv21,
         29: chainer_model.conv22,
         30: chainer_model.conv23}
    for index, val in d.items():
        if index != 30:
            cba = model.models[index]
            copy_conv_bn_activ(cba, val)
            if index == 0:
                val.conv.W.data /= 255
        else:
            conv = model.models[index][0]
            if conv.weight.shape == val.W.data.shape:
                val.W.data[:] = conv.weight
                val.b.data[:] = conv.bias
            else:
                print('skipping the last layer src_shape={} dst_shape={}'.format(
                    conv.weight.shape, val.W.data.shape))

    chainer.serializers.save_npz(args.output, chainer_model)

    # Check correctness
    # import torch
    # model.eval()
    # chainer.config.train = False
    # x = np.load('img.npy')
    # a =torch.autograd.Variable(torch.Tensor(x))
    # chainer_out = chainer_model(x)
    # out = model.forward(a)
    # print(chainer_out.shape)
    # print(out.shape)
    # print(chainer_out[0, 0, :5, :5])
    # print(out[0, 0, :5, :5].data.numpy())
