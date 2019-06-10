import numpy as np
import unittest

from siam_rpn.general.eval_sot_vot import eval_sot_vot

# How the data is generated
# all_tracker_trajs = np.load(
#     'all_tracker_trajs.npz', allow_pickle=True)['all_tracker_trajs']
# 
# new_pred_statuses = []
# new_pred_bboxes = []
# for tracker_traj in all_tracker_trajs:
#     the_pred_bboxes = []
#     the_pred_statuses = []
#     for v in tracker_traj[0]:
#         if len(v) == 1:
#             the_pred_bboxes.append(None) 
#             if v[0] == 1:
#                 the_pred_statuses.append(0)
#             elif v[0] == 2:
#                 the_pred_statuses.append(2)
#             elif v[0] == 0:
#                 the_pred_statuses.append(3)
#         else:
#             the_pred_statuses.append(1)
#             x, y, w, h = v
#             bbox = np.array([[y, x, y + h, x + w]], dtype=np.float32)
#             the_pred_bboxes.append(bbox)
#     new_pred_statuses.append(the_pred_statuses)
#     new_pred_bboxes.append(the_pred_bboxes)
# 
# np.savez(
#     'siamrpn_r50_l234_dwxcorr.npz', pred_bboxes=new_pred_bboxes,
#     pred_statuses=new_pred_statuses, gt_polys=gt_polys, sizes=sizes)

class TestEvalVOT(unittest.TestCase):

    def setUp(self):
        data = np.load('eval_siamrpn_r50_l234_dwxcorr.npz', allow_pickle=True)
        self.pred_bboxes = data['pred_bboxes']
        self.pred_statuses = data['pred_statuses']
        self.gt_polys = data['gt_polys']
        self.sizes = data['sizes']

    def test(self):
        result = eval_sot_vot(
            self.pred_bboxes, self.pred_statuses,
            self.gt_polys, self.sizes
            )

        np.testing.assert_almost_equal(result['accuracy'],  0.602063329, decimal=6)
        np.testing.assert_almost_equal(result['robustness'], 0.23880876, decimal=6)
        np.testing.assert_almost_equal(result['eao'],  0.4128777468, decimal=4)
