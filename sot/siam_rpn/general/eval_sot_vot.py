import numpy as np

from siam_rpn.general.vot_tracking_dataset import get_axis_aligned_bb

try:
    from toolkit.utils.region import vot_overlap as orig_vot_overlap
except:
    pass


def calc_failure_frame_ids(statuses):
    # on one video
    failure = [i for i, status in enumerate(statuses)
               if status == 2]
    return failure


def calculate_expected_overlap(fragments, fweights):
    max_len = fragments.shape[1]
    expected_overlaps = np.zeros((max_len), np.float32)
    expected_overlaps[0] = 1

    # TODO Speed Up 
    for i in range(1, max_len):
        mask = np.logical_not(np.isnan(fragments[:, i]))
        if np.any(mask):
            fragment = fragments[mask, 1:i+1]
            seq_mean = np.sum(fragment, 1) / fragment.shape[1]
            expected_overlaps[i] = np.sum(seq_mean *
                fweights[mask]) / np.sum(fweights[mask])
    return expected_overlaps


def vot_overlap(pred_bbox, gt_poly, size):
    H, W = size
    if not isinstance(pred_bbox, list):
        pred_bb = pred_bbox[0]
        pred_bb = [
            pred_bb[1], pred_bb[0],
            pred_bb[3] - pred_bb[1],
            pred_bb[2] - pred_bb[0]]
    else:
        pred_bb = pred_bbox

    gt_ply = gt_poly[0]
    if len(gt_ply.shape) == 2:
        assert gt_ply.shape == (4, 2)
        gt_ply = gt_ply[:, ::-1].reshape(-1)
    else:
        gt_ply = [
            gt_ply[1], gt_ply[0],
            gt_ply[3] - gt_ply[1],
            gt_ply[2] - gt_ply[0]]
    return orig_vot_overlap(pred_bb, gt_ply, (W, H))


def vot_overlaps(pred_bboxes, pred_statuses, gt_polys, size):
    assert len(pred_bboxes) == len(gt_polys)
    overlaps = []
    for pred_bbox, pred_status, gt_poly in zip(pred_bboxes, pred_statuses, gt_polys):
        if pred_status != 1:
            pred_bbox = [0]
        # if pred_status == 0:
        #     pred_bbox = 1
        # elif pred_status == 2:
        #     pred_bbox = 2
        # elif pred_status == 3:
        #     pred_bbox = 0
        overlaps.append(vot_overlap(pred_bbox, gt_poly, size))
    return overlaps


def calculate_accuracy(
    pred_bboxes, pred_statuses, gt_polys,
    burnin=0, bound=None):
    """Caculate accuracy socre as average overlap over the entire sequence

    Args:
        burnin: number of frames that have to be ignored after the failure
        bound: bounding region

    Return:
        acc: average overlap
        overlaps: per frame overlaps
    """
    
    pred_statuses = np.array(pred_statuses)
    n_img = len(pred_bboxes)
    if burnin > 0:
        valid = np.where(pred_statuses == 0)[0]

        for i in valid:
            for j in range(burnin):
                if i + j < n_img:
                    pred_statuses[i + j] = 3  # ignore

    assert len(pred_bboxes) == len(gt_polys)
    # min_len = min(len(pred_trajectory_), len(gt_trajectory))
    overlaps = vot_overlaps(
        pred_bboxes, pred_statuses, gt_polys, bound)

    acc = np.nanmean(overlaps) if len(overlaps) > 0 else 0
    return acc, overlaps


def eval_sot_vot(pred_bboxes, pred_statuses, gt_polys, sizes):
    out1 = calc_overlaps_and_failure_frame_ids(
        pred_bboxes, pred_statuses, gt_polys, sizes, burnin=10)
    accuracy, robustness = calc_sot_vot_ar(*out1)

    out = calc_overlaps_and_failure_frame_ids(
        pred_bboxes, pred_statuses, gt_polys, sizes, burnin=0)
    eao, expected_overlaps = calc_sot_vot_eao(*out)
    return {
        'accuracy': accuracy,
        'robustness': robustness,
        'eao': eao,
        'expected_overlaps': expected_overlaps,
    }


def calc_overlaps_and_failure_frame_ids(
        pred_bboxes, pred_statuses, gt_polys, sizes, burnin=0):
    all_overlaps = []
    all_failure_frame_ids = []
    for the_pred_bboxes, the_pred_statuses, the_gt_polys, the_sizes in zip(
        pred_bboxes, pred_statuses, gt_polys, sizes):
        _, overlaps = calculate_accuracy(
            the_pred_bboxes, the_pred_statuses, the_gt_polys,
            burnin=burnin, bound=the_sizes[0])
        all_overlaps.append(overlaps)
        all_failure_frame_ids.append(calc_failure_frame_ids(the_pred_statuses))
    return all_overlaps, all_failure_frame_ids


def calc_sot_vot_ar(all_overlaps, all_failure_frame_ids):
    flat_all_overlaps = []
    for overlaps in all_overlaps:
        for v in overlaps:
            flat_all_overlaps.append(v)
    acc = np.nanmean(flat_all_overlaps)

    all_n_failure = [len(frame_ids) for frame_ids in all_failure_frame_ids]
    total_length = sum([len(overlaps) for overlaps in all_overlaps]) 
    percentage_failure = np.sum(all_n_failure) / total_length  # [0, 1]
    robustness = percentage_failure * 100
    return acc, robustness


def calc_sot_vot_eao(all_overlaps, all_failure_frame_ids, mode='vot2018'):
    if mode == 'vot2018':
        low = 100
        high = 356
        # peak = 160

    video_lengths = [len(overlaps) for overlaps in all_overlaps]

    n_video = len(all_overlaps)

    n_fragment = sum([len(frame_ids) + 1 for frame_ids in all_failure_frame_ids])
    max_len = max([len(overlaps) for overlaps in all_overlaps])
    seq_weight = 1 / (n_video + 1e-10)

    fweights = np.ones((n_fragment,)) * np.nan 
    fragments = np.ones((n_fragment, max_len)) * np.nan

    n_skip = 5
    seq_counter = 0
    for video_length, failure_frame_ids, overlaps in zip(
            video_lengths, all_failure_frame_ids, all_overlaps):
        if len(failure_frame_ids) > 0:
            points = [frame_id + n_skip for frame_id in failure_frame_ids
                      if frame_id + n_skip <= len(overlaps)]
            points = [0] + points
            for i in range(len(points)):
                start = points[i]
                if i != len(points) - 1:
                    end = points[i + 1]
                    fragment = overlaps[start:end + 1]
                    fragments[seq_counter, :] = 0
                else:
                    fragment = overlaps[start:]
                fragment = np.array(fragment)
                fragment[np.isnan(fragment)] = 0
                fragments[seq_counter, :len(fragment)] = fragment

                fweights[seq_counter] = seq_weight

                seq_counter += 1
        else:
            # no failure happend in this video
            assert len(overlaps) <= max_len
            fragments[seq_counter, :len(overlaps)] = overlaps

            fweights[seq_counter] = seq_weight
            seq_counter += 1

    expected_overlaps = calculate_expected_overlap(fragments, fweights)

    weight = np.zeros((len(expected_overlaps),))
    weight[low - 1:high - 1 + 1] = 1
    is_valid = np.logical_not(np.isnan(expected_overlaps))
    eao = (
        expected_overlaps[is_valid] * weight[is_valid]).sum() / weight[is_valid].sum()
    return eao, expected_overlaps
