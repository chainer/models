import argparse

import os

import chainer

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.chainer_experimental.datasets.sliceable import TupleDataset
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.utils import write_image

from dataset_utils import get_sbd_augmented_voc
from dataset_utils import SimpleDoesItTransform
from voc_semantic_segmentation_with_bbox_dataset import \
    VOCSemanticSegmentationWithBboxDataset
from model import get_pspnet_resnet50


def predict_all(model, dataset):
    labels = []
    for i in range(len(dataset)):
        if i % 100 == 0:
            print('{}/{}'.format(i, len(dataset)))
        img = dataset[i][0]
        label = model.predict([img])[0]
        labels.append(label)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=6)
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--out', type=str, default='result')
    args = parser.parse_args()

    # Model
    model = get_pspnet_resnet50(len(voc_semantic_segmentation_label_names))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model)
    raw_train_data = get_sbd_augmented_voc()
    train_data = raw_train_data

    debug = True

    for i in range(10):
        if i < 6:
            lr = 0.001
        else:
            lr = 0.0001

        out = os.path.join(args.out, 'epoch_{0:02d}'.format(i))
        train_one_epoch(model, train_data, lr, args.gpu, args.batchsize, out)
        print('finished training a epoch')

        labels = predict_all(model, raw_train_data)
        if debug:
            for i in range(len(labels)):
                label = labels[i]
                write_image(label[None],
                            os.path.join(out, 'image_{0:05d}.png'.format(i)))
        train_data = TupleDataset(raw_train_data, labels)


def train_one_epoch(model, train_data, lr, gpu, batchsize, out):
    train_model = PixelwiseSoftmaxClassifier(model)
    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        train_model.to_gpu()  # Copy the model to the GPU
    log_trigger = (0.1, 'epoch')
    validation_trigger = (1, 'epoch')
    end_trigger = (1, 'epoch')

    train_data = TransformDataset(
        train_data, ('img', 'label_map'), SimpleDoesItTransform(model.mean))
    val = VOCSemanticSegmentationWithBboxDataset(
        split='val').slice[:, ['img', 'label_map']]

    # Iterator
    train_iter = iterators.MultiprocessIterator(train_data, batchsize)
    val_iter = iterators.MultiprocessIterator(
        val, 1, shuffle=False, repeat=False, shared_mem=100000000)

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0001))

    # Updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=out)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss'], x_key='iteration',
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['validation/main/miou'], x_key='iteration',
            file_name='miou.png'))

    trainer.extend(extensions.snapshot_object(
        model, filename='snapshot.npy'),
        trigger=end_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
         'main/loss', 'validation/main/miou',
         'validation/main/mean_class_accuracy',
         'validation/main/pixel_accuracy']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        SemanticSegmentationEvaluator(
            val_iter, model,
            voc_semantic_segmentation_label_names),
        trigger=validation_trigger)
    trainer.run()


if __name__ == '__main__':
    main()
