import argparse
import math
import chainer
import chainer.links as L
from chainer import training
from chainer import function
from chainer.training import triggers
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.datasets import TransformDataset
from chainer.training import extensions
from models.vgg import VGG16
from models.preresnet import PreResNet110
from models.wide_resnet import WideResNet28x10


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
    parser.add_argument('--se', action='store_true',
                        help='snapshot ensemble usage flag')
    parser.add_argument('--se_cycle', type=int, default=5,
                        help='split the training process into N cycles, '
                             'each of which starts with a large LR')

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

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(args.lr_init, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.wd))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    stop_trigger = (args.epoch, 'epoch')

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Learning rate adjustment (this function is called every epoch)
    def baseline_lr_schedule(trainer):
        epoch = trainer.updater.epoch
        t = epoch / args.epoch

        factor = 1.0
        if t >= 0.5:
            factor = 0.1
        elif t >= 0.75:
            factor = 0.01
        trainer.updater.get_optimizer('main').lr = factor * args.lr_init

    total_iter = len(train) * args.epoch // args.batchsize
    cycle_iter = math.floor(total_iter / args.se_cycle)

    # Learning rate adjustment (this function is called every epoch)
    def cycle_lr_schedule(trainer):
        iter = trainer.updater.iteration
        lr = args.lr_init * 0.5
        lr *= math.cos(math.pi * ((iter - 1) % cycle_iter) / cycle_iter) + 1
        trainer.updater.get_optimizer('main').lr = lr

    # Set up extentions
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    if args.se:
        trainer.extend(extensions.snapshot(), trigger=(cycle_iter, 'iteration'))
        trainer.extend(cycle_lr_schedule,
                       trigger=triggers.IntervalTrigger(1, 'iteration'))
    else:
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
        trainer.extend(baseline_lr_schedule,
                       trigger=triggers.IntervalTrigger(1, 'epoch'))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.LogReport())
    cols = ['epoch', 'lr', 'main/loss', 'main/accuracy',
            'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']
    trainer.extend(extensions.PrintReport(cols))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
