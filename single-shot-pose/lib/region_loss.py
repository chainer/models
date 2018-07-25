from __future__ import division
import numpy as np

import chainer.functions as F

from .ssp import rpoints_to_points


def corner_confidences9(gt_corners, pr_corners,
                        th=80, sharpness=2,
                        img_W=640, img_H=480):
    xp = chainer.cuda.get_array_module(gt_corners)
    assert pr_corners.shape[:2] == (9, 2)
    assert gt_corners.shape[:2] == (9, 2)
    if gt_corners.ndim == 2:
        gt_corners = gt_corners[:, :, None, None]
    if pr_corners.ndim == 2:
        pr_corners = pr_corners[:, :, None, None]
        squeeze = True
    else:
        squeeze = False
    _, _, H, W = pr_corners.shape

    dist = (gt_corners - pr_corners)
    dist = dist.reshape(9, 2, -1)
    dist[:, 0] *= img_W
    dist[:, 1] *= img_H

    dist = xp.sqrt(xp.sum(dist ** 2, axis=1))
    mask = (dist < th).astype(np.float32)
    conf = xp.exp(sharpness * (1 - dist / th)) - 1

    conf0 = xp.exp(sharpness * 1) - 1
    conf /= conf0
    conf *= mask
    mean_conf = xp.mean(conf, axis=0).reshape(H, W)
    if squeeze:
        mean_conf = mean_conf[0, 0]
    return mean_conf


def create_target(pred_corners, gt_points, noobject_scale=0.1,
                  object_scale=5, 
                  sil_thresh=0.6):
    xp = chainer.cuda.get_array_module(gt_points)
    B, _, _, H, W = pred_corners.shape
    B = len(gt_points)

    conf_mask = noobject_scale * xp.ones((B, H, W), dtype=np.bool)
    coord_mask = xp.zeros((B, H, W), dtype=np.bool)
    gt_confs = xp.zeros((B, H, W), dtype=np.float32)
    gt_rpoints = xp.zeros((B, 9, 2, H, W), dtype=np.float32)

    for b, (pred_corner, gt_point) in enumerate(
            zip(pred_corners, gt_points)):
        conf = xp.zeros((H, W), dtype=np.float32)
        for gt_pt in gt_point:
            conf = xp.maximum(
                conf, corner_confidences9(gt_pt, pred_corner))
        conf_mask[b, conf > sil_thresh] = 0

    for b, (pred_corner, gt_point) in enumerate(
            zip(pred_corners, gt_points)):
        for gt_pt in gt_point:
            # NOTE: 0 <= gt_pt[0, 1] - gi0 < 1
            gi0 = int(xp.floor(gt_pt[0, 1] * W))
            gj0 = int(xp.floor(gt_pt[0, 0] * H))
            conf = corner_confidences9(
                gt_pt, pred_corner[:, :, gi0, gj0])

            conf_mask[b, gi0, gj0] = object_scale
            coord_mask[b, gi0, gj0] = 1
            gt_confs[b, gi0, gj0] = conf
            gt_rpoints[b, :, 0, gi0, gj0] = gt_pt[:, 0] * W - gj0
            gt_rpoints[b, :, 1, gi0, gj0] = gt_pt[:, 1] * H - gi0
    return gt_rpoints, gt_confs, coord_mask, conf_mask


def region_loss(output, gt_points):
    xp = chainer.cuda.get_array_module(output)
    # rpoints lie in [0, feat_size]
    # points lie in [0, 1]
    # anchors = [0.1067, 0.9223]
    B, C, H, W = output.shape
    n_class = C - 19

    det_confs = F.sigmoid(output[:, 18])
    cls_conf = F.softmax(output[:, 19:19 + n_class])
    rpoints = output[:, :18].reshape(B, 9, 2, H, W)
    rpoints0 = F.sigmoid(rpoints[:, 0])
    rpoints = F.concat(
        (rpoints0[:, None], rpoints[:, 1:]), axis=1)
    rpoints_data = rpoints.data

    points_data = rpoints_to_points(rpoints_data)
    gt_rpoints, gt_confs, coord_mask, conf_mask = create_target(
        points_data, gt_points)

    coord_loss = F.sum(
        coord_mask[:, None, None] * (rpoints - gt_rpoints) ** 2) / 2
    conf_loss = F.sum(conf_mask * (det_confs - gt_confs) ** 2) / 2
    loss = coord_loss + conf_loss
    return loss


if __name__ == '__main__':
    import chainer
    output = np.load('loss_input.npy')
    target = np.load('target_input.npy')


    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
    anno = target.reshape(-1, 50, 21)
    points = []
    for i in range(anno.shape[0]):
        anno_i = anno[i][:truths_length(anno[i])]
        # normalized
        points.append(anno_i[:, 1:19].reshape(-1, 9, 2).astype(np.float32))

    loss = region_loss(chainer.Variable(output), points)



    # import pickle
    # import torch
    # with open('gt_corners.pkl', 'rb') as f:
    #     gt_corners = pickle.load(f)
    # with open('pr_corners.pkl', 'rb') as f:
    #     pr_corners = pickle.load(f)
    # conf = corner_confidence9(np.array(gt_corners), np.array(pr_corners))

    gt_corners = np.load(
        'gt_corners.npy').reshape(9, 2, 13, 13)
    pr_corners = np.load('pr_corners.npy').reshape(9, 2, 13, 13)
    output = np.load('mean_conf.npy')
    mean_conf = corner_confidences9(gt_corners, pr_corners)
    np.testing.assert_almost_equal(mean_conf.reshape(-1), output)
