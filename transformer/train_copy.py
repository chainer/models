import argparse

import chainer

from chains.copy_transformer import CopyTransformer
from datasets.copy_dataset import CopyDataset
from evaluation.copy_transformer_eval_function import CopyTransformerEvaluationFunction, CopyTransformerEvaluator
from hooks.noam_hook import NoamOptimizer
from updater.copy_transformer_updater import CopyTransformerUpdater


def main(args):
    # prepare the datasets and iterators
    train_dataset = CopyDataset(args.vocab_size)

    train_iter = chainer.iterators.MultithreadIterator(train_dataset, args.batch_size)

    # build the network we want to train
    net = CopyTransformer(args.vocab_size, train_dataset.max_len, train_dataset.start_symbol)

    # build the optimizer
    optimizer = chainer.optimizers.Adam(alpha=0, beta1=0.9, beta2=0.98, eps=1e-9)
    optimizer.setup(net)
    # this hook is very important! Without it, the training won't converge!
    optimizer.add_hook(
        NoamOptimizer(4000, 2, net.transformer_size)
    )

    # create our custom updater that computes the loss and updates the params of the network
    updater = CopyTransformerUpdater(train_iter, optimizer, device=args.gpu)

    # init the trainer
    trainer = chainer.training.Trainer(updater, (args.epochs, 'epoch'), out='train')
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'loss', 'train/accuracy', 'part_accuracy', 'accuracy']))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    # create the evaluator
    eval_function = CopyTransformerEvaluationFunction(net, args.gpu)
    trainer.extend(CopyTransformerEvaluator({}, net, device=args.gpu, eval_func=eval_function))

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the transformer under a copy task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-g", "--gpu", default=-1, type=int, help="gpu device to use (negative value indicates cpu)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("-v", "--vocab-size", type=int, default=101, help="number of different digits that we train the copy on")

    args = parser.parse_args()
    main(args)

