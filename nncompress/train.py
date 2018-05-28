# -*- coding: utf-8 -*-
import argparse
import os

import chainer
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions
from chainer.training.triggers import MinValueTrigger

from compressor.data_processor import DataProcessor
from compressor.net import EmbeddingCompressor
from compressor.resource import Resource
from compressor.subfuncs import set_random_seed, save_non_embed_npz


def main():
    parser = argparse.ArgumentParser(description='Embedding Compressor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--iter', '-i', dest='iteration', type=int, default=200000,
                        help='Number of iterations')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--optimizer', '-O', dest='optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'], help='Type of optimizer')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--M' '-M', dest='n_codebooks', type=int, default=32,
                        help='Number of Codebooks')
    parser.add_argument('--K' '-K', dest='n_centroids', type=int, default=16,
                        help='Number of Centroids (Number of vectors in each codebook)')
    parser.add_argument('--tau', dest='tau', type=float, default=1.0,
                        help='Tau value in Gumbel-softmax')

    # Arguments for the dataset / vocabulary path
    parser.add_argument('--input-matrix', dest='input_matrix', required=True,
                        help='path to the matrix (npy)')

    # Random Seed
    parser.add_argument('--seed', default=0, type=int, help='Seed for Random Module')

    # Arguments for directory
    parser.add_argument('--out', '-o', default='./result', type=os.path.abspath,
                        help='Directory to output the result')
    parser.add_argument('--dir-prefix', dest='dir_prefix', default='model', type=str, help='Prefix of the output dir')
    args = parser.parse_args()
    set_random_seed(args.seed, args.gpu)

    resource = Resource(args, train=True)
    resource.dump_git_info()
    resource.dump_command_info()
    resource.dump_python_info()
    resource.dump_chainer_info()
    resource.save_config_file()

    logger = resource.logger

    dataset = DataProcessor(resource.log_name)
    dataset.load_embed_matrix(args.input_matrix)
    train_data = dataset.load_data('train')
    valid_data = dataset.load_data('dev')
    model = EmbeddingCompressor(
        n_vocab=dataset.embed_matrix.shape[0],
        embed_dim=dataset.embed_matrix.shape[1],
        n_codebooks=args.n_codebooks,
        n_centroids=args.n_centroids,
        tau=args.tau,
        embed_mat=dataset.embed_matrix
    )

    if args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    else:
        optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(model)
    logger.info('Optimizer is set to [{}]'.format(args.optimizer))
    model.embed_mat.disable_update()  # call this after optimizer.setup()
    logger.info('Updating Embedding Layer is Disabled')

    # Send model to GPU (according to the arguments)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    train_iter = SerialIterator(dataset=train_data, batch_size=args.batchsize, shuffle=True)

    updater = training.updater.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=resource.output_dir)

    short_term = (1000, 'iteration')

    dev_iter = SerialIterator(valid_data, args.batchsize, repeat=False)
    trainer.extend(
        extensions.Evaluator(dev_iter, model, device=args.gpu), trigger=short_term)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.LogReport(trigger=short_term, log_name='chainer_report_iteration.log'),
                   trigger=short_term, name='iteration')
    trainer.extend(extensions.LogReport(trigger=short_term, log_name='chainer_report_epoch.log'), trigger=short_term,
                   name='epoch')
    trainer.extend(extensions.snapshot_object(model, 'iter_{.updater.iteration}.npz', savefun=save_non_embed_npz),
                   trigger=MinValueTrigger('validation/main/loss', short_term))

    entries = ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/maxp', 'validation/main/maxp']
    trainer.extend(extensions.PrintReport(entries=entries, log_report='iteration'), trigger=short_term)
    trainer.extend(extensions.PrintReport(entries=entries, log_report='epoch'), trigger=short_term)

    logger.info('Start training...')
    trainer.run()
    logger.info('Training complete!!')
    resource.dump_duration()


if __name__ == "__main__":
    main()
