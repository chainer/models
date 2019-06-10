import argparse
import numpy as np

import chainer

from siam_rpn.general.eval_sot_vot import eval_sot_vot
from siam_rpn.siam_rpn import SiamRPN
from siam_rpn.siam_rpn_tracker import SiamRPNTracker
from siam_rpn.siam_mask_tracker import SiamMaskTracker
from siam_rpn.general.vot_tracking_dataset import VOTTrackingDataset

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from chainer import iterators

from siam_rpn.general.predictor_with_gt import PredictorWithGT


def collate_images_from_same_video(data, used_ids=None):
    imgs = data.slice[:, 'img']
    polys = data.slice[:, 'poly']
    video_ids = data.slice[:, 'video_id']
    frame_ids = data.slice[:, 'frame_id']
    if used_ids is None:
        used_ids = np.unique(video_ids)
        np.sort(used_ids)

    videos = []
    video_polys = []
    for video_id in used_ids:
        indices = np.where(video_ids == video_id)[0]
        the_frame_ids = list(frame_ids.slice[indices])
        assert all(list(the_frame_ids) == np.arange(len(the_frame_ids)))
        videos.append(imgs.slice[indices])
        video_polys.append(polys[indices])
    return videos, video_polys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()

    data = VOTTrackingDataset('data')

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

    videos, video_polys = collate_images_from_same_video(
        data, used_ids=None)
    video_dataset = chainer.datasets.TupleDataset(videos, video_polys)

    it = iterators.SerialIterator(video_dataset, 1, False, False)

    in_values, out_values, rest_values = apply_to_iterator(
        PredictorWithGT(tracker, mask=args.mask), it,
        n_input=2, hook=ProgressHook(len(video_dataset)))
    # delete unused iterators explicitly
    imgs, video_polys = in_values
    pred_bboxes, pred_statuses, sizes = out_values
    del imgs

    video_polys = list(video_polys)
    pred_bboxes = list(pred_bboxes)
    pred_statuses = list(pred_statuses)
    sizes = list(sizes)
    np.savez(
        'eval_sot_out.npz',
         pred_bboxes=pred_bboxes, pred_statuses=pred_statuses,
         gt_polys=video_polys, sizes=sizes)
    result = eval_sot_vot(pred_bboxes, pred_statuses, video_polys, sizes)
    print(result['eao'], result['accuracy'], result['robustness'])
