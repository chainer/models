import argparse
import math
import glob
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer import function
from chainer import serializers
from chainer.training import triggers
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.dataset import convert
from chainer.training import extensions
from chainer.backends import cuda
from models.vgg import VGG16
from models.preresnet import PreResNet110
from models.wide_resnet import WideResNet28x10


def concat_arrays(arrays):
    # Convert `arrays` to numpy.ndarray or cupy.ndarray
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate(arrays)


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar100',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--model', '-m', default='VGG16',
                        help='The model to use: VGG16 or PreResNet110'
                             ' or WideResNet28x10')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory from which read the snapshot files')

    args = parser.parse_args()

    if args.dataset.lower() == 'cifar10':
        print('Using CIFAR10 dataset')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset.lower() == 'cifar100':
        print('Using CIFAR100 dataset')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    print('Using %s model' % args.model)
    if args.model == 'VGG16':
        model_cls = VGG16
    elif args.model == 'PreResNet110':
        model_cls = PreResNet110
    elif args.model == 'WideResNet28x10':
        model_cls = WideResNet28x10
    else:
        raise RuntimeError('Invalid model choice.')

    model = model_cls(class_labels)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    t = np.array([data[1] for data in test], np.int32)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        t = cuda.cupy.array(t)

    def predict(model, test_iter):
        probs = []
        test_iter.reset()

        for batch in test_iter:
            in_arrays = convert.concat_examples(batch, args.gpu)

            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                y = model(in_arrays[0])
                prob = chainer.functions.softmax(y)
                probs.append(prob.data)
        return concat_arrays(probs)

    # gather each model's softmax outputs
    results = []
    for snapshot_path in glob.glob(args.out + '/*snapshot*'):
        serializers.load_npz(snapshot_path, model,
                             path='updater/model:main/predictor/')
        y = predict(model, test_iter)
        acc = F.accuracy(y, t)
        results.append(y[None])
        print('accuracy:', acc.data)

    # compute the average
    results = concat_arrays(results)
    y = results.mean(axis=0)

    acc = F.accuracy(y, t)
    print('-'*50)
    print('ensemble accuray:', acc.data)


if __name__ == '__main__':
    main()
