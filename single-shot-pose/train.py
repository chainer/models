from __future__ import division

import argparse
import numpy as np
import random

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset

from lib.linemod_dataset import LinemodDataset
from lib.linemod_dataset import linemod_object_diameters
from lib.mesh import MeshPly
from lib.projected_3d_bbox_evaluator import Projected3dBboxEvaluator
from lib.region_loss import region_loss
from lib.ssp import SSPYOLOv2
from lib.transforms import Transform
from lib.utils import get_linemod_intrinsics


class TrainChain(chainer.Chain):

    def __init__(self, model, iterator=None, conf_loss_scale=1):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.iterator = iterator
        self.conf_loss_scale = conf_loss_scale

    def __call__(self, imgs, points, labels):
        n_seen = int(self.iterator.epoch_detail * len(self.iterator.dataset))
        if n_seen < 400*32:
            size = 13*32
        else:
            size = (random.randint(0,7) + 13)*32
        # elif n_seen < 800*32:
        #     size = (random.randint(0,7) + 13)*32
        # elif n_seen < 1200*32:
        #     size = (random.randint(0,9) + 12)*32
        # elif n_seen < 1600*32:
        #     size = (random.randint(0,11) + 11)*32
        # elif n_seen < 2000*32:
        #     size = (random.randint(0,13) + 10)*32
        # elif n_seen < 2400*32:
        #     size = (random.randint(0,15) + 9)*32
        # elif n_seen < 3000*32:
        #     size = (random.randint(0,17) + 8)*32
        # else: # self.seen < 20000*64:
        #     size = (random.randint(0,19) + 7)*32

        imgs, original_Ws, original_Hs = self.model.prepare(imgs, img_W=size, img_H=size)
        for point, original_W, original_H in zip(points, original_Ws, original_Hs):
            # Rescale according to image resize
            point[:, :, 0] *= (size / original_W)
            point[:, :, 1] *= (size / original_H)
            # Rescale to [0, 1]
            point[:, :, 0] /= size
            point[:, :, 1] /= size

        imgs = self.xp.array(imgs)
        points = [self.xp.array(point) for point in points]
        labels = [self.xp.array(label) for label in labels]
        model_outputs = self.model(imgs)
        point_loss, conf_loss = region_loss(model_outputs, points)
        loss = point_loss + self.conf_loss_scale * conf_loss

        chainer.reporter.report(
            {'loss': loss, 'n_seen': n_seen,
             'loss/point': point_loss, 'loss/conf': conf_loss},
            self)
        return loss


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--pretrained-model', default='', type=str,
                        help='Output directory')
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--conf-loss-scale', '-c', type=float, default=1)
    parser.add_argument('object')
    args = parser.parse_args()

    train_data = LinemodDataset('.', obj_name=args.object, split='train', return_msk=True)
    train_data = TransformDataset(train_data, ('img', 'point', 'label'), Transform())
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=args.batchsize, n_processes=None, shared_mem=100000000)
    test_data = LinemodDataset('.', obj_name=args.object, split='test')
    test_iter = chainer.iterators.SerialIterator(
        test_data, 1, repeat=False, shuffle=False)

    model = SSPYOLOv2()
    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model)
    train_chain = TrainChain(model, train_iter)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name != 'beta' and param.name != 'gamma':
            param.update_rule.add_hook(
                chainer.optimizer.WeightDecay(0.0005))

    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=converter)

    trainer = training.Trainer(
        updater, (700, 'epoch'), out=args.out)
    # trainer.extend(
    #     extensions.ExponentialShift('lr', 10),
    #     trigger=chainer.triggers.ManualScheduleTrigger([50], 'epoch'))
    mesh = MeshPly('LINEMOD/{}/{}.ply'.format(args.object, args.object))
    vertex      = np.c_[
        np.array(mesh.vertices),
        np.ones((len(mesh.vertices), 1))]
    intrinsics = get_linemod_intrinsics()
    trainer.extend(
        Projected3dBboxEvaluator(
            test_iter, model, vertex, intrinsics,
            diam=linemod_object_diameters[args.object]),
        trigger=(10, 'epoch'))

    log_interval = 1, 'epoch'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/n_seen',
         'main/loss', 'main/loss/point', 'main/loss/conf',
         'validation/main/point2d_acc',
         'validation/main/proj_acc',
         'validation/main/trans_rot_acc',
         'validation/main/point3d_acc',
         'validation/main/mean_proj_error',
         'validation/main/mean_point2d_error',
         ]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.epoch}'),
        trigger=(100, 'epoch'))
    trainer.run()
