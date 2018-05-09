import argparse
import random
from functools import partial
import chainer
import chainer.links as L
from chainer import training
from chainer import function
from chainer.training import triggers
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.datasets import TransformDataset
from chainer.training import extensions
import cupy as cp
from models.vgg import VGG16
from models.preresnet import PreResNet110
from models.wide_resnet import WideResNet28x10


class SwaEvaluator(extensions.Evaluator):
    default_name = "swa"


def transform(in_data, train=True, crop_size=32, padding=4):
    img, label = in_data
    img = img.copy()
    xp = cp.get_array_module(img)

    # Random flip & crop
    if train:
        # Random flip
        if random.randint(0, 1):
            img = img[:, :, ::-1]

        # Random crop
        pad_img = xp.pad(img, [(0, 0), (padding, padding), (padding, padding)],
                         'constant')
        C, H, W = pad_img.shape
        top = random.randint(0, H - crop_size - 1)
        left = random.randint(0, W - crop_size - 1)
        bottom = top + crop_size
        right = left + crop_size
        img = pad_img[:, top:bottom, left:right]

    # Normalize
    mean = xp.array([0.485, 0.456, 0.406])
    std = xp.array([0.229, 0.224, 0.225])
    img -= mean[:, None, None]
    img /= std[:, None, None]

    return img, label


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar100',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--model', '-m', default='VGG16',
                        help='The model to use: VGG16 or PreResNet110'
                             ' or WideResNet28x10')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--lr_init', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--swa', action='store_true',
                        help='swa usage flag')
    parser.add_argument('--swa_start', type=float, default=161,
                        help='SWA start epoch number')
    parser.add_argument('--swa_lr', type=float, default=0.05,
                        help='SWA LR')
    parser.add_argument('--swa_c_epochs', type=int, default=1,
                        help='SWA model collection frequency length in epochs')

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

    model = L.Classifier(model_cls(class_labels))

    if args.swa:
        swa_model = L.Classifier(model_cls(class_labels))
        swa_n = 0

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        if args.swa:
            swa_model.to_gpu()

    # Data augmentation / preprocess
    train = TransformDataset(train, partial(transform, train=True))
    test = TransformDataset(test, partial(transform, train=False))

    optimizer = chainer.optimizers.MomentumSGD(args.lr_init, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.wd))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    swa_train_iter = chainer.iterators.SerialIterator(
        train, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    stop_trigger = (args.epoch, 'epoch')

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Learning rate adjustment (this function is called every epoch)
    def lr_schedule(trainer):
        epoch = trainer.updater.epoch
        t = epoch / (args.swa_start if args.swa else args.epoch)
        lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01

        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        trainer.updater.get_optimizer('main').lr = factor * args.lr_init

    # The main function for SWA (this function is called every epoch)
    def avg_weight(trainer):
        epoch = trainer.updater.epoch
        if args.swa and (epoch + 1) >= args.swa_start and \
                (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            nonlocal swa_n
            # moving average
            alpha = 1.0 / (swa_n + 1)
            for param1, param2 in zip(swa_model.params(), model.params()):
                param1.data *= (1.0 - alpha)
                param1.data += param2.data * alpha
            swa_n += 1

    # This funtion is called before evaluating SWA model
    # for fixing batchnorm's running mean and variance
    def fix_swa_batchnorm(evaluator):
        # Check batchnorm layer
        bn_flg = False
        for l in swa_model.links():
            if type(l) == L.normalization.batch_normalization.BatchNormalization:
                bn_flg = True
                break

        # Fix batchnorm's running mean and variance
        if bn_flg:
            swa_train_iter.reset()
            with chainer.using_config('train', True):
                for batch in swa_train_iter:
                    in_arrays = evaluator.converter(batch, evaluator.device)
                    with function.no_backprop_mode():
                        swa_model(*in_arrays)

    # Set up extentions
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),
                   trigger=(5, 'epoch'))
    if args.swa:
        eval_points = [x for x in range(args.epoch + 1)
                       if x > args.swa_start and x % 5 == 0]
        trainer.extend(SwaEvaluator(test_iter, swa_model, device=args.gpu, eval_hook=fix_swa_batchnorm),
                       trigger=triggers.ManualScheduleTrigger(eval_points, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(lr_schedule, trigger=triggers.IntervalTrigger(1, 'epoch'))
    trainer.extend(avg_weight, trigger=triggers.IntervalTrigger(1, 'epoch'))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.LogReport())
    cols = ['epoch', 'lr', 'main/loss', 'main/accuracy',
            'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']
    if args.swa:
        cols = cols[:-1] + ['swa/main/loss', 'swa/main/accuracy'] + cols[-1:]
    trainer.extend(extensions.PrintReport(cols))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
