import argparse

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import links as L

import dataset
from model import RelationNetwork


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=str, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum number of epochs to train')
    parser.add_argument('--val-iter', type=int, default=100,
                        help='Run validation every N-th epoch')
    parser.add_argument('--log-iter', type=int, default=100,
                        help='Log every N-th iteration')
    parser.add_argument('--snapshot-iter', type=int, default=10000,
                        help='Model snapshot every N-th epoch')
    parser.add_argument('--n-val-questions', type=int, default=4000,
                        help='Number of questions for validation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--sort-of-clevr-path', type=str,
                        default='sort_of_clevr.pkl',
                        help='Sort-of-CLEVR pickle file')
    parser.add_argument('--out', type=str, default='result',
                        help='Output directory')
    args = parser.parse_args()

    dataset, clevr = dataset.get_sort_of_clevr(args.sort_of_clevr_path)

    train = dataset[:-args.n_val_questions]
    val = dataset[-args.n_val_questions:]

    train_iter = iterators.SerialIterator(train, args.batch_size)
    val_iter = chainer.iterators.SerialIterator(
        val, args.batch_size, repeat=False, shuffle=False)

    relation_network = RelationNetwork(len(clevr.vocab))
    model = L.Classifier(relation_network)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer=optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, out=args.out, stop_trigger=(args.max_epochs, 'epoch'))

    trainer.extend(
        extensions.Evaluator(val_iter, target=model, device=args.gpu),
        trigger=(args.val_iter, 'iteration')
    )
    trainer.extend(
        extensions.LogReport(
            ['main/loss', 'validation/main/loss', 'main/accuracy',
             'validation/main/accuracy'],
            trigger=(args.log_iter, 'iteration')
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(args.log_iter, 'iteration'), file_name='loss.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            trigger=(args.log_iter, 'iteration'), file_name='accuracy.png'
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss',
             'validation/main/loss', 'main/accuracy',
             'validation/main/accuracy']
        ),
        trigger=(args.log_iter, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(
            relation_network, 'snapshot_{.updater.iteration}'),
        trigger=(args.snapshot_iter, 'iteration')
    )
    trainer.extend(extensions.ProgressBar())

    trainer.run()
