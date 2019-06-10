import os
import argparse

import torch
import numpy as np

import chainer
import chainercv

from siam_rpn.siam_rpn import SiamRPN

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='model name')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()
    single_scale = args.mask

    chainer_model = SiamRPN(multi_scale=not single_scale, mask=args.mask)
    chainer.serializers.load_npz(args.pretrained_model, chainer_model)

    cfg.merge_from_file(args.config)
    # cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval()

    # Check model
    chainer.config.train = False
    extractor = chainer_model.extractor
    neck = chainer_model.neck
    rpn = chainer_model.rpn

    if os.path.exists('template.npy'):
        z = np.load('template.npy')
        x = np.load('track.npy')
    else:
        z = np.random.uniform(size=(1, 3, 255, 255)).astype(np.float32)
        x = z
        raise ValueError
    outs = [v.cpu().detach().numpy() 
            for v in model.backbone(torch.FloatTensor(z))]
    # [print(out.shape) for out in outs]
    if args.mask:
        extractor.pick = ('conv1', 'res2', 'res3', 'res4')
    else:
        extractor.pick = ('res3', 'res4', 'res5')
    c_out = extractor(z)
    for i in range(len(c_out)):
        np.testing.assert_almost_equal(
            c_out[i].data, outs[i], decimal=5)

    model.template(torch.FloatTensor(z))
    chainer_model.template(z)

    np.testing.assert_almost_equal(
            model.zf.detach().numpy(), chainer_model.zs.data, decimal=5)

    t_out = model.track(torch.FloatTensor(x))
    c_out = chainer_model.track(x)
    t_conf = t_out['cls'].detach().numpy()
    t_loc = t_out['loc'].detach().numpy()
    np.testing.assert_almost_equal(t_conf, c_out[0].data, decimal=5)
    np.testing.assert_almost_equal(t_loc, c_out[1].data, decimal=5)
    if args.mask:
        t_mask = t_out['mask'].detach().numpy()
        np.testing.assert_almost_equal(t_mask, c_out[2].data, decimal=5)

    pos = (12, 11)
    t_out = model.mask_refine(pos).detach().numpy()
    c_out = chainer_model.refine_mask(pos).data
    np.testing.assert_almost_equal(t_out, c_out, decimal=5)
    print('test done')
