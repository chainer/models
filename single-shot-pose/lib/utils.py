from __future__ import division
import numpy as np

import cv2


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


def get_linemod_intrinsics():
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K


def calc_angular_distance(gt_rot, pr_rot):
    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff) 
    return np.rad2deg(np.arccos((trace-1.0)/2.0))


def compute_projection(points_3D, transformation, internal_calibration):
    points_3D = points_3D.T
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d.T


def get_3d_corners(vertices):
    vertices = vertices.T
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate(
        (corners.T, np.ones((1,8))), axis=0).T
    return corners


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                              # points_2D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)

    R, _ = cv2.Rodrigues(R_exp)
    return R, t
