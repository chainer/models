from __future__ import division

import numpy as np
from utils import corner_confidences9
from utils import calc_angular_distance
from utils import compute_projection
from utils import get_3d_corners
from utils import pnp


def eval_projected_3d_bbox(
        pred_points, pred_labels, pred_scores,
        gt_points, gt_labels, vertex, intrinsics, diam):
    errors = calc_projected_3d_bbox_error(
        pred_points, pred_labels, pred_scores,
        gt_points, gt_labels, vertex, intrinsics)

    px_thresh = 5
    eps = 1e-5

    denom = len(errors['point2d']) + eps
    point2d_acc = len(np.where(errors['point2d'] <= px_thresh)[0]) * 100 / denom
    proj_acc = len(np.where(errors['proj'] <= px_thresh)[0]) * 100 / denom
    trans_rot_acc = len(
        np.where((errors['trans'] <= 0.05) & (errors['rot'] <= 5))[0]) * 100 / denom
    point3d_acc = len(np.where(errors['point_3d'] <= (diam * 0.1))[0]) * 100 / denom

    return {'point2d_acc': point2d_acc,
            'proj_acc': proj_acc,
            'trans_rot_acc': trans_rot_acc,
            'point3d_acc': point3d_acc,
            'mean_proj_error': np.mean(errors['proj']),
            'mean_point2d_error': np.mean(errors['point2d'])}


def calc_projected_3d_bbox_error(
        pred_points, pred_labels, pred_scores,
        gt_points, gt_labels, vertex, intrinsics):
    gt_pnt_3d = get_3d_corners(vertex)
    point_2d_errors = []
    trans_errors = []
    rot_errors = []
    proj_errors = []
    point_3d_errors = []
    for pred_point, pred_score, gt_point in zip(
            pred_points, pred_scores, gt_points):
        # only one ground truth
        assert gt_point.shape[0] == 1
        gt_pnt = gt_point[0]
        pred_pnt = pred_point[np.argmax(pred_score)]

        points_3d = np.concatenate(
            (np.zeros((1, 3)), gt_pnt_3d[:, :3]), axis=0)
        gt_R, gt_t = pnp(points_3d, gt_pnt, intrinsics)
        pred_R, pred_t = pnp(points_3d, pred_pnt, intrinsics)
        gt_Rt = np.concatenate((gt_R, gt_t), axis=1)
        pred_Rt = np.concatenate((pred_R, pred_t), axis=1)

        gt_proj_pnt = compute_projection(vertex, gt_Rt, intrinsics)
        pred_proj_pnt = compute_projection(vertex, pred_Rt, intrinsics)

        gt_transformed_pnt = gt_Rt.dot(vertex.T).T
        pred_transformed_pnt = pred_Rt.dot(vertex.T).T

        # 2D point error
        point_2d_errors.append(
            np.mean(np.linalg.norm(pred_pnt - gt_pnt, axis=1)))
        # 3D pose error
        trans_errors.append(np.sqrt(np.sum(np.square(gt_t - pred_t))))
        rot_errors.append(calc_angular_distance(gt_R, pred_R))
        # projected 2D error 
        proj_errors.append(np.mean(
            np.linalg.norm(gt_proj_pnt - pred_proj_pnt, axis=1)))
        # 3D point error
        point_3d_errors.append(np.mean(
            np.linalg.norm(
                gt_transformed_pnt - pred_transformed_pnt, axis=1)))
    return {'point2d': np.array(point_2d_errors),
            'trans': np.array(trans_errors),
            'rot': np.array(rot_errors),
            'proj': np.array(proj_errors),
            'point_3d': np.array(point_3d_errors)}


if __name__ == '__main__':
    # import pickle
    # with open('a.pkl', 'rb') as f:
    #     all_points = pickle.load(f)

    # all_points = [np.array(all_points[0])]
    # pred_points = np.array([all_points[0][:, :18]]).reshape(1, -1, 9, 2)
    # pred_points[:, :, :, 0] *= 640
    # pred_points[:, :, :, 1] *= 480
    # pred_scores = [all_points[0][:, 18]]


    # truths = np.load('target.npy')
    # truths = truths.reshape(1, -1, 21)[:, :1]
    # gt_points = truths[:, :, 1:19].reshape(1, 1, 9, 2)
    # gt_points[:, :, :, 0] *= 640
    # gt_points[:, :, :, 1] *= 480
    gts = np.load('data/box_gts.npy')
    gt_points = gts[:, :18].reshape(-1, 1, 9, 2)
    gt_points[:, :, :, 0] *= 640
    gt_points[:, :, :, 1] *= 480
    preds = np.load('data/boxes_list.npy')
    pred_points = []
    pred_scores = []
    for pred in preds:
        pred_point = []
        pred_score = []
        for pr in pred:
            pred_point.append(pr[:18])
            pred_score.append(pr[18])
        pred_point = np.array(pred_point).reshape(-1, 9, 2)
        pred_point[:, :, 0] *= 640
        pred_point[:, :, 1] *= 480
        pred_points.append(np.array(pred_point).reshape(-1, 9, 2))
        pred_scores.append(np.array(pred_score))


    from utils import get_camera_intrinsic
    from mesh import MeshPly

    intrinsics = get_camera_intrinsic()
    mesh = MeshPly('../LINEMOD/ape/ape.ply')
    vertex      = np.c_[
        np.array(mesh.vertices),
        np.ones((len(mesh.vertices), 1))]

    np.savez('linemod_ape.npz',
             pred_points=pred_points, pred_scores=pred_scores,
             gt_points=gt_points, diam=0.103, vertex=vertex)

    # errors = calc_projected_3d_bbox_error(
    #     pred_points, None, pred_scores,
    #     gt_points, None, vertex, intrinsics)
    result = eval_projected_3d_bbox(
        pred_points, None, pred_scores,
        gt_points, None, vertex, intrinsics, diam=0.103)
