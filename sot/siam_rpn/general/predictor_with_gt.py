import numpy as np
from siam_rpn.general.vot_tracking_dataset import get_axis_aligned_bb

from .eval_sot_vot import vot_overlap


class PredictorWithGT(object):

    def __init__(self, tracker, mask=False):
        self.tracker = tracker
        self.mask = mask

    def __call__(self, videos, video_gt_polys):
        video_pred_bboxes = []
        video_pred_statuses = []
        video_sizes = []
        for imgs, gt_polys in zip(videos, video_gt_polys):
            out = self.run_for_one_video(imgs, gt_polys)
            video_pred_bboxes.append(out[0])
            video_pred_statuses.append(out[1])
            video_sizes.append(out[2])
        return video_pred_bboxes, video_pred_statuses, video_sizes

    def run_for_one_video(self, imgs, gt_polys):
        pred_bboxes = []
        statuses = []
        sizes = []

        start_idx = 0
        found_thresh = 0
        for i, (img, gt_poly) in enumerate(zip(imgs, gt_polys)):
            n = len(gt_poly)
            gt_bbox = get_axis_aligned_bb(gt_poly[0])[None]
            sizes.append(img.shape[1:])
            if i == start_idx:
                self.tracker.init(img, gt_bbox)
                pred_bboxes.append(None)
                # 0: initial
                # 1: found
                # 2: lost
                # 3: ignore
                statuses.append(np.zeros((n,), dtype=np.int32))
            elif i > start_idx:
                if self.mask:
                    pred_bbox, _, _, pred_poly = self.tracker.track(img)
                    overlap = vot_overlap(pred_poly, gt_bbox, img.shape[1:])
                else:
                    pred_bbox, _ = self.tracker.track(img)
                    overlap = vot_overlap(pred_bbox, gt_bbox, img.shape[1:])
                if overlap > found_thresh:
                    pred_bboxes.append(pred_bbox)
                    statuses.append(np.ones((n,), dtype=np.int32))
                else:
                    pred_bboxes.append(None)
                    start_idx = i + 5  # skip 5 frames
                    statuses.append(2 * np.ones((n,), dtype=np.int32))
            elif i < start_idx:
                # ignore
                pred_bboxes.append(None)
                statuses.append(3 * np.ones((n,), dtype=np.int32))
        return pred_bboxes, statuses, sizes
