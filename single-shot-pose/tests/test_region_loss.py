import unittest

import numpy as np

import chainer

import sys
sys.path.append('..')
from lib.region_loss import region_loss


class TestRegionLoss(unittest.TestCase):

    gt_x_loss = 1.399209
    gt_y_loss = 2.2095947
    gt_conf_loss = 2.120961

    def test_cpu(self):
        data = np.load('test_region_loss.npz')

        point_loss, conf_loss = region_loss(data['output'], data['gt_points'])

        np.testing.assert_almost_equal(
            point_loss.data, self.gt_x_loss + self.gt_y_loss, decimal=5)
        np.testing.assert_almost_equal(
            conf_loss.data, self.gt_conf_loss, decimal=5)

    def test_gpu(self):
        data = np.load('test_region_loss.npz')

        gt_points = [data['gt_points'][0], data['gt_points'][0]]
        gt_points = [chainer.cuda.to_gpu(point) for point in gt_points]
        outputs = np.concatenate((data['output'], data['output']), axis=0)

        point_loss, conf_loss = region_loss(
            chainer.cuda.to_gpu(outputs), gt_points)

        np.testing.assert_almost_equal(
            chainer.cuda.to_cpu(point_loss.data), self.gt_x_loss + self.gt_y_loss, decimal=5)
        np.testing.assert_almost_equal(
            chainer.cuda.to_cpu(conf_loss.data), self.gt_conf_loss, decimal=5)
