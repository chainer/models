import argparse
import torch
import chainer

from siam_rpn.siam_rpn import SiamRPN


def copy_conv(l, key, val, bias=False, flip=False):
    print(key)
    if bias:
        l.b.data[:] = val.cpu().numpy()
    else:
        if flip:
            l.W.data[:] = val.cpu().numpy()[:, ::-1]
        else:
            l.W.data[:] = val.cpu().numpy()

def copy_bn(l, key, val):
    print(key)
    if key[-6:] == 'weight':
        l.gamma.data[:] = val.cpu().numpy()
    elif key[-4:] == 'bias':
        l.beta.data[:] = val.cpu().numpy()
    elif key[-12:] == 'running_mean':
        l.avg_mean.data[:] = val.cpu().numpy()
    elif key[-11:] == 'running_var':
        l.avg_var.data[:] = val.cpu().numpy()

def backbone_parse(chainer_model, key, val):
    # print(key)
    if 'conv1' == key[:5]:
        copy_conv(chainer_model.conv1.conv, key, val, flip=True)
    if 'bn1' == key[:3]:
        copy_bn(chainer_model.conv1.bn, key, val)
    
    if 'layer' == key[:5]:
        s_keys = key.split('.')
        m = key[5]
        res_l = getattr(chainer_model, 'res{}'.format(int(m) + 1))
        if '0' == s_keys[1]:
            base_l = getattr(res_l, 'a')
        else:
            base_l = getattr(res_l, 'b{}'.format(s_keys[1]))

        if 'conv' in s_keys[2]:
            n = s_keys[2][len('conv'):]
            conv = getattr(
                base_l, 'conv{}'.format(n))
            copy_conv(conv.conv, key, val)
        elif 'bn' in s_keys[2]:
            n = s_keys[2][len('bn'):]
            conv = getattr(
                base_l, 'conv{}'.format(n))
            copy_bn(conv.bn, key, val)
        elif 'downsample' == s_keys[2]:
            if s_keys[3] == '0':
                l = base_l.residual_conv.conv
                copy_conv(l, key, val)
            elif s_keys[3] == '1':
                l = base_l.residual_conv.bn
                copy_bn(l, key, val)

def neck_parse(chainer_model, key, val, single_scale=False):
    s_keys = key.split('.')
    if single_scale:
        if s_keys[0][:10] == 'downsample':
            l = getattr(chainer_model, 'downsample')
            if '0' == s_keys[2]:
                copy_conv(l.conv.conv, key, val)
            elif '1' == s_keys[2]:
                copy_bn(l.conv.bn, key, val)
    else:
        if s_keys[0][:10] == 'downsample':
            m = s_keys[0][10]
            l = getattr(chainer_model, 'downsample{}'.format(int(m) + 1))
            if '0' == s_keys[2]:
                copy_conv(l.conv.conv, key, val)
            elif '1' == s_keys[2]:
                copy_bn(l.conv.bn, key, val)


def depthwise_x_corr_parse(chainer_model, s_keys, val):
    l = chainer_model
    if s_keys[0] == 'conv_kernel':
        l = l.conv_kernel
    elif s_keys[0] == 'conv_search':
        l = l.conv_search
    elif s_keys[0] == 'head':
        if s_keys[1] == '0':
            l = l.conv_head1
        elif s_keys[1] == '1':
            l = l.conv_head1
        elif s_keys[1] == '3':
            l = l.conv_head2

    if s_keys[1] == '0':
        copy_conv(l.conv, key, val)
    elif s_keys[1] == '1':
        copy_bn(l.bn, key, val)
    elif s_keys[1] == '3':
        if s_keys[2] == 'weight':
            copy_conv(l, key, val)
        elif s_keys[2] == 'bias':
            copy_conv(l, key, val, bias=True)

def rpn_parse(chainer_model, key, val, single_scale=False):

    def _one_rpn_parse(l):
        if s_keys[1] == 'loc':
            l = l.loc
        elif s_keys[1] == 'cls':
            l = l.conf
        depthwise_x_corr_parse(l, s_keys[2:], val)

    s_keys = key.split('.')
    if s_keys[0][:10] == 'cls_weight':
        print(key)
        chainer_model.conf_weight.data[:] = val.cpu()
    elif s_keys[0][:10] == 'loc_weight':
        print(key)
        chainer_model.loc_weight.data[:] = val.cpu()
    elif s_keys[0][:3] == 'rpn':
        m = s_keys[0][3]
        l = getattr(chainer_model, 'rpn{}'.format(int(m) + 1))
        _one_rpn_parse(l)
    else:
        if single_scale:
            l = chainer_model
            s_keys = [''] + s_keys
            _one_rpn_parse(l)


def mask_refine_parse(chainer_model, key, val):
    s_keys = key.split('.')
    if s_keys[0] in ['v0', 'v1', 'v2', 'h2', 'h1', 'h0']:
        l = getattr(chainer_model, s_keys[0])

        if s_keys[1] == '0':
            l = l[0].conv
            if s_keys[2] == 'weight':
                copy_conv(l, key, val, bias=False)
            if s_keys[2] == 'bias':
                copy_conv(l, key, val, bias=True)
        if s_keys[1] == '2':
            l = l[1].conv
            if s_keys[2] == 'weight':
                copy_conv(l, key, val, bias=False)
            if s_keys[2] == 'bias':
                copy_conv(l, key, val, bias=True)
    elif s_keys[0] in ['deconv', 'post0', 'post1', 'post2']:
        l = getattr(chainer_model, s_keys[0])
        if s_keys[1] == 'weight':
            copy_conv(l, key, val, bias=False)
        if s_keys[1] == 'bias':
            copy_conv(l, key, val, bias=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--out', default='chainer_model.npz', type=str)

    args = parser.parse_args()

    single_scale = args.mask

    chainer_model = SiamRPN(multi_scale=not single_scale, mask=args.mask)

    weight = torch.load(args.snapshot)
    for key, val in weight.items():
        if 'backbone.' == key[:len('backbone.')]:
            key = key[len('backbone.'):]
            backbone_parse(chainer_model.extractor, key, val)
        elif 'neck.' == key[:len('neck.')]:
            key = key[len('neck.'):]
            neck_parse(chainer_model.neck, key, val, single_scale)
        elif 'rpn_head.' == key[:len('rpn_head.')]:
            key = key[len('rpn_head.'):]
            rpn_parse(chainer_model.rpn, key, val, single_scale)
        elif 'mask_head.' == key[:len('mask_head.')]:
            key = key[len('mask_head.'):]
            s_keys = key.split('.')
            depthwise_x_corr_parse(chainer_model.mask_head, s_keys, val)
        elif 'refine_head.' == key[:len('refine_head.')]:
            key = key[len('refine_head.'):]
            mask_refine_parse(chainer_model.mask_refine, key, val)
        
    chainer.serializers.save_npz(args.out, chainer_model)
    print('done conversion')
