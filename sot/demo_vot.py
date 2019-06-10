import argparse

import chainer
import matplotlib.pyplot as plt
import numpy as np

from siam_rpn.siam_rpn import SiamRPN
from siam_rpn.siam_rpn_tracker import SiamRPNTracker
from siam_rpn.siam_mask_tracker import SiamMaskTracker

from siam_rpn.general.vot_tracking_dataset import VOTTrackingDataset
from siam_rpn.general.vis_bbox_video import vis_bbox_video
from siam_rpn.general.vis_instance_segmentation_video import vis_instance_segmentation_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--video-id', type=int, default=0)
    parser.add_argument('--out', type=str, default='demo_video.mp4')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    if args.mask:
        model = SiamRPN(multi_scale=False, mask=True)
        chainer.serializers.load_npz(args.pretrained_model, model)
        tracker = SiamMaskTracker(model)
    else:
        model = SiamRPN()
        chainer.serializers.load_npz(args.pretrained_model, model)
        tracker = SiamRPNTracker(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        tracker.to_gpu()

    data = VOTTrackingDataset('data')
    indices = np.where(np.array(data.slice[:, 'video_id']) == args.video_id)[0]

    first_img, first_bbox = data.slice[:, ['img', 'bbox']][indices[0]]
    tracker.init(first_img, first_bbox)
    pred_bboxes = [first_bbox]
    imgs = [first_img]

    mask_shape = (len(first_bbox),) + first_img.shape[1:]
    pred_masks = [np.zeros(mask_shape, dtype=np.bool)]
    for i in range(len(indices) - 1):
        img = data.slice[:, 'img'][indices[i + 1]]
        imgs.append(img)
        if args.mask:
            bbox, best_score, mask, _ = tracker.track(img)
            pred_masks.append(mask)
        else:
            bbox, best_score = tracker.track(img)[:2]
        pred_bboxes.append(bbox)

    if args.mask:
        ani = vis_instance_segmentation_video(imgs, pred_masks)
        ani.save(args.out, writer='ffmpeg', fps=30)
    else:
        ani = vis_bbox_video(imgs, pred_bboxes)
        ani.save(args.out, writer='ffmpeg', fps=30)
