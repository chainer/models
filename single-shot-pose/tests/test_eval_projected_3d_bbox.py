import unittest

import numpy as np

import sys
sys.path.append('../lib')
from eval_projected_3d_bbox import eval_projected_3d_bbox_single
from utils import get_linemod_intrinsics


class TestEvalProjected3dBbox(unittest.TestCase):

    def test(self):
        # download validation input/ouptut
        # https://drive.google.com/open?id=1gjTKzGptvvCu5ElhaUv6alX91alIL4IJ
        data = np.load('linemod_ape.npz')

        result = eval_projected_3d_bbox_single(
            data['pred_points'], data['pred_scores'],
            data['gt_points'], data['vertex'],
            get_linemod_intrinsics(), diam=data['diam'])

        ######################################################################
        # output of valid.py
        ######################################################################
        # Acc using 5 px 2D Projection = 94.48%
        # Acc using 10% threshold - 0.0103 vx 3D Transformation = 28.00%
        # Acc using 5 cm 5 degree metric = 51.90%
        # Mean 2D pixel error is 2.833865, Mean vertex error is 0.029564,
        # mean corner error is 4.058189
        # Translation error: 0.029320 m, angle error: 5.689846 degree,
        # pixel error:  2.833865 pix
        print(result)
        np.testing.assert_almost_equal(
            result['proj_acc'], 94.48, decimal=2)
        np.testing.assert_almost_equal(
            result['point3d_acc'], 28.00, decimal=2)
        np.testing.assert_almost_equal(
            result['trans_rot_acc'], 51.90, decimal=2)
        np.testing.assert_almost_equal(
            result['mean_proj_error'], 2.8338654, decimal=6)
        np.testing.assert_almost_equal(
            result['mean_point2d_error'], 4.058189, decimal=6)
        
