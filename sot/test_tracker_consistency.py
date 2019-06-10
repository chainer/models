import os
import argparse

import torch
import numpy as np

import chainer
import chainercv

from siam_rpn.siam_rpn import SiamRPN
from siam_rpn.siam_mask_tracker import SiamMaskTracker
from siam_rpn.siam_rpn_tracker import SiamRPNTracker

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from siam_rpn.general.vot_tracking_dataset import VOTTrackingDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='model name')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()
    single_scale = args.mask

    c_model = SiamRPN(multi_scale=not single_scale, mask=args.mask)
    chainer.serializers.load_npz(args.pretrained_model, c_model)
    if args.mask:
        c_tracker = SiamMaskTracker(c_model)
    else:
        c_tracker = SiamRPNTracker(c_model)

    cfg.merge_from_file(args.config)
    cfg.CUDA = False
    # device = torch.device('cuda' if cfg.CUDA else 'cpu')
    t_model = ModelBuilder()
    t_model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    t_model.eval()
    t_tracker = build_tracker(t_model)

    data = VOTTrackingDataset('data')
    indices = np.where(np.array(data.slice[:, 'video_id']) == 10)[0]

    first_img, first_bbox = data.slice[:, ['img', 'bbox']][indices[0]]

    bb = first_bbox[0]
    hwc = first_img.transpose(1,2,0)[:, :, ::-1]
    xyhw = [bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0]]

    t_tracker.init(hwc, xyhw)
    c_tracker.init(first_img, first_bbox)
    if args.mask:
        np.testing.assert_almost_equal(
                c_tracker.model.zs.data,
                t_tracker.model.zf.detach().numpy(), decimal=5)
    else:
        for i in range(len(c_tracker.model.zs)):
            np.testing.assert_almost_equal(
                    c_tracker.model.zs[i].data,
                    t_tracker.model.zf[i].detach().numpy(), decimal=5)

    for i in range(80):
        print('checking {}'.format(i))
        img = data.slice[:, 'img'][indices[i + 1]]
        t_out = t_tracker.track(img.transpose(1,2,0)[:, :, ::-1])
        c_out = c_tracker.track(img)
        t_bb = [
                t_out['bbox'][1],
                t_out['bbox'][0],
                t_out['bbox'][1] + t_out['bbox'][3],
                t_out['bbox'][0] + t_out['bbox'][2]]

        np.testing.assert_almost_equal(t_bb, c_out[0][0], decimal=4)
        if args.mask:
            np.testing.assert_almost_equal(t_out['polygon'], c_out[3], decimal=4)
            # np.testing.assert_almost_equal(t_out['nn_mask'], c_out[-1], decimal=4)
            # np.testing.assert_almost_equal(t_out['mask'], c_out[2], decimal=4)
