import argparse
import numpy as np

import chainer
from chainercv.utils import read_image
from chainercv.visualizations import vis_image

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from lib.eval_projected_3d_bbox import eval_projected_3d_bbox_single
from lib.linemod_dataset import LinemodDataset
from lib.linemod_dataset import linemod_object_diameters
from lib.mesh import MeshPly
from lib.ssp import SSPYOLOv2
from lib.utils import get_linemod_intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('object')
    args = parser.parse_args()

    model = SSPYOLOv2()
    chainer.serializers.load_npz(args.pretrained_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    test = LinemodDataset('.', obj_name=args.object, split='test')
    it = chainer.iterators.SerialIterator(
        test, batch_size=1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, it, hook=ProgressHook(len(test)))
    del in_values
    points, labels, scores = out_values
    gt_points, gt_labels = rest_values

    intrinsics = get_linemod_intrinsics()
    mesh = MeshPly('LINEMOD/{}/{}.ply'.format(args.object, args.object))
    vertex      = np.c_[
        np.array(mesh.vertices),
        np.ones((len(mesh.vertices), 1))]
    result = eval_projected_3d_bbox_single(
        points, scores,
        gt_points, vertex, intrinsics,
        diam=linemod_object_diameters[args.object])
    print('Acc using 5 px 2D projection = {0:.2f}'.format(result['proj_acc']))
    print('Acc using 10% threshold: 3D transformation = {0:.2f}'.format(
        result['point3d_acc']))
    print('Acc using 5 cm 5 degree metric = {0:.2f}'.format(
        result['trans_rot_acc']))
    print('Mean projected 2D pixel error = {0:.4f}'.format(
        result['mean_proj_error']))
    print('Mean 2D pixel error = {0:.4f}'.format(
        result['mean_point2d_error']))
