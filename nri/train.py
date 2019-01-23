#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import re
import time

import chainer
import chainer.functions as F
import numpy as np
from chainer import backends
from chainer import computational_graph
from chainer import dataset
from chainer import iterators
from chainer import optimizers
from chainer import serializers

import matplotlib.pyplot as plt
from nri.datasets import graph_dataset
from nri.models import decoders
from nri.models import encoders


def create_result_dir(out):
    dname = time.strftime('%Y-%m-%d_%H-%M-%S_0', time.gmtime())
    dname = os.path.join(out, dname)
    i = 0
    while os.path.exists(dname):
        i = int(dname.split('_')[-1]) + 1
        dname = '_'.join(dname.split('_')[:-1]) + '_{}'.format(i)
    os.makedirs(dname)
    return dname


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Terminate after 10 iteration')
    parser.add_argument('--out', '-o', type=str, default='results')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO, etc.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--train-loc-npy', type=str, default='data/springs/loc_train_springs5.npy')
    parser.add_argument('--train-vel-npy', type=str, default='data/springs/vel_train_springs5.npy')
    parser.add_argument('--train-edges-npy', type=str, default='data/springs/edges_train_springs5.npy')

    parser.add_argument('--valid-loc-npy', type=str, default='data/springs/loc_valid_springs5.npy')
    parser.add_argument('--valid-vel-npy', type=str, default='data/springs/vel_valid_springs5.npy')
    parser.add_argument('--valid-edges-npy', type=str, default='data/springs/edges_valid_springs5.npy')

    parser.add_argument('--test-loc-npy', type=str, default='data/springs/loc_test_springs5.npy')
    parser.add_argument('--test-vel-npy', type=str, default='data/springs/vel_test_springs5.npy')
    parser.add_argument('--test-edges-npy', type=str, default='data/springs/edges_test_springs5.npy')

    parser.add_argument('--timesteps', type=int, default=49)
    parser.add_argument('--feature-dims', type=int, default=4)
    parser.add_argument('--edge-types', type=int, default=2)
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--skip-first', action='store_true', default=False,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')

    parser.add_argument('--encoder', type=str, default='mlp')
    parser.add_argument('--encoder-hidden', type=int, default=256)
    parser.add_argument('--encoder-dropout', type=float, default=0.0)
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')

    parser.add_argument('--decoder', type=str, default='mlp')
    parser.add_argument('--decoder-hidden', type=int, default=256)
    parser.add_argument('--decoder-dropout', type=float, default=0.0)
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')

    parser.add_argument('--batch-size', '-b', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')

    args = parser.parse_args()
    args.out = create_result_dir(args.out)
    args.factor = not args.no_factor

    # set up logging to file - see previous section for more details
    logging.basicConfig(
        level=args.logging_level,
        format='%(asctime)s (%(name)s) [%(levelname)s]: %(message)s',
        filename=os.path.join(args.out, 'log.txt'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    formatter = logging.Formatter('%(asctime)s (%(name)s) [%(levelname)s]: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return args


def get_encoder(encoder_type, timesteps, feature_dims, encoder_hidden, edge_types, encoder_dropout, factor):
    if encoder_type == 'mlp':
        encoder = encoders.MLPEncoder(
            n_in=timesteps * feature_dims,
            n_hid=encoder_hidden,
            n_out=edge_types,
            do_prob=encoder_dropout,
            factor=factor
        )
    elif encoder_type == 'cnn':
        encoder = encoders.CNNEncoder(
            n_in=feature_dims,
            n_hid=encoder_hidden,
            n_out=edge_types,
            do_prob=encoder_dropout,
            factor=factor
        )
    else:
        raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

    return encoder


def get_decoder(decoder_type, timesteps, feature_dims, decoder_hidden, edge_types, decoder_dropout, skip_first):
    if decoder_type == 'mlp':
        decoder = decoders.MLPDecoder(
            n_in_node=feature_dims,
            edge_types=edge_types,
            msg_hid=decoder_hidden,
            msg_out=decoder_hidden,
            n_hid=decoder_hidden,
            do_prob=decoder_dropout,
            skip_first=skip_first
        )
    elif decoder_type == 'rnn':
        decoder = decoders.RNNDecoder(
            n_in_node=feature_dims,
            edge_types=edge_types,
            n_hid=decoder_hidden,
            do_prob=decoder_dropout,
            skip_first=skip_first
        )
    else:
        raise ValueError('Unsupported decoder type: {}'.format(decoder_type))

    return decoder


def get_iterator(loc_npy, vel_npy, edges_npy):
    ds = graph_dataset.GraphDataset(loc_npy, vel_npy, edges_npy)
    return iterators.SerialIterator(ds, args.batch_size)


def get_sender_receiver(num_nodes):
    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)

    # Split receiver and sender node IDs
    rel_send, rel_rec = np.where(off_diag)

    # Make them 1-hot vectors
    rel_send = np.eye(len(np.unique(rel_send)))[rel_send]
    rel_rec = np.eye(len(np.unique(rel_rec)))[rel_rec]

    return rel_send.astype(np.float32), rel_rec.astype(np.float32)


def get_edge_accuracy(edge_probs, edge_labels):
    num_correct = np.sum(np.argmax(edge_probs, axis=-1) == edge_labels)
    return float(num_correct / edge_probs.shape[0] / edge_probs.shape[1])


def put_log(epoch, loss_nll, loss_kl, edge_accuracy, node_mse, split='train'):
    logger = logging.getLogger(__name__)
    logger.info('Epoch [{}]: {}\tloss_nll: {:.4f}\tloss_kl: {:.4f}\tedge_accuracy:{:.4f}'
                '\tnode_mse: {:.2e}'.format(
                    split, epoch, loss_nll, loss_kl, edge_accuracy, node_mse))


def get_nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * (F.log(2 * preds.xp.array(preds.xp.pi, dtype=preds.dtype)) + \
                       F.log(preds.xp.array(variance, preds.dtype)))
        neg_log_p += const
    ret = F.sum(neg_log_p) / (target.shape[0] * target.shape[1])
    return ret


def get_kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * F.log(preds + eps)
    if add_const:
        const = F.log(preds.xp.array(num_edge_types, dtype=preds.dtype))
        kl_div += const
    return F.sum(kl_div) / (num_atoms * preds.shape[0])


def train(
        iterator, gpu, encoder, decoder, enc_optim, dec_optim, rel_send, rel_rec, edge_types,
        temp, prediction_steps, var, out, benchmark, lr_decay, gamma):
    iter_i = 0
    edge_accuracies = []
    node_mses = []
    nll_train = []
    kl_train = []

    logger = logging.getLogger(__name__)

    while True:
        inputs = iterator.next()
        node_features, edge_labels = dataset.concat_examples(inputs, device=gpu)

        # logits: [batch_size, num_edges, edge_types]
        logits = encoder(node_features, rel_send, rel_rec)  # inverse func. of softmax
        edges = F.gumbel_softmax(logits, tau=temp, axis=2)
        edge_probs = F.softmax(logits, axis=2)
        # edges, edge_probs: [batch_size, num_edges, edge_types]

        if isinstance(decoder, decoders.MLPDecoder):
            output = decoder(
                node_features, edges, rel_rec, rel_send, prediction_steps)
        elif isinstance(decoder, decoders.RNNDecoder):
            output = decoder(
                node_features, edges, rel_rec, rel_send, 100,
                burn_in=True,
                burn_in_steps=args.timesteps - args.prediction_steps)

        target = node_features[:, :, 1:, :]
        num_nodes = node_features.shape[1]

        loss_nll = get_nll_gaussian(output, target, var)
        loss_kl = get_kl_categorical_uniform(edge_probs, num_nodes, edge_types)

        loss = loss_nll + loss_kl

        nll_train.append(float(loss_nll.array))
        kl_train.append(float(loss_kl.array))

        edge_accuracy = get_edge_accuracy(logits.array, edge_labels)
        edge_accuracies.append(edge_accuracy)

        node_mse = float(F.mean_squared_error(output, target).array)
        node_mses.append(node_mse)

        encoder.cleargrads()
        decoder.cleargrads()
        loss.backward()
        enc_optim.update()
        dec_optim.update()

        # Exit after 10 iterations when benchmark mode is ON
        iter_i += 1
        if benchmark:
            put_log(iterator.epoch, np.mean(nll_train), np.mean(kl_train),
                    np.mean(edge_accuracies), np.mean(node_mses))
            if iter_i == 10:
                exit()

        if iterator.is_new_epoch:
            break

    if not os.path.exists(os.path.join(out, 'graph.dot')):
        with open(os.path.join(out, 'graph.dot'), 'w') as o:
            g = computational_graph.build_computational_graph([loss])
            o.write(g.dump())

    if iterator.is_new_epoch:
        put_log(iterator.epoch, np.mean(nll_train), np.mean(kl_train), np.mean(edge_accuracies), np.mean(node_mses))
        serializers.save_npz(os.path.join(out, 'encoder_epoch-{}.npz'.format(iterator.epoch)), encoder)
        serializers.save_npz(os.path.join(out, 'decoder_epoch-{}.npz'.format(iterator.epoch)), decoder)
        serializers.save_npz(os.path.join(out, 'enc_optim_epoch-{}.npz'.format(iterator.epoch)), enc_optim)
        serializers.save_npz(os.path.join(out, 'dec_optim_epoch-{}.npz'.format(iterator.epoch)), dec_optim)

        if iterator.epoch % lr_decay == 0:
            enc_optim.alpha *= gamma
            dec_optim.alpha *= gamma
            logger.info('alpha of enc_optim: {}'.format(enc_optim.alpha))
            logger.info('alpha of dec_optim: {}'.format(dec_optim.alpha))


def valid(iterator, gpu, encoder, decoder, rel_send, rel_rec, edge_types, temp, var):
    nll_valid = []
    kl_valid = []
    edge_accuracies = []
    node_mses = []

    chainer.config.train = False
    chainer.config.enable_backprop = False

    while True:
        inputs = iterator.next()
        node_features, edge_labels = dataset.concat_examples(inputs, device=gpu)

        # logits: [batch_size, num_edges, edge_types]
        logits = encoder(node_features, rel_send, rel_rec)  # inverse func. of softmax
        edges = F.gumbel_softmax(logits, tau=temp, axis=2)
        edge_probs = F.softmax(logits, axis=2)
        # edges, edge_probs: [batch_size, num_edges, edge_types]

        # validation output uses teacher forcing
        output = decoder(node_features, edges, rel_rec, rel_send, 1)

        target = node_features[:, :, 1:, :]
        num_nodes = node_features.shape[1]

        loss_nll = get_nll_gaussian(output, target, var)
        loss_kl = get_kl_categorical_uniform(edge_probs, num_nodes, edge_types)

        nll_valid.append(float(loss_nll.array))
        kl_valid.append(float(loss_kl.array))

        edge_accuracy = get_edge_accuracy(logits.array, edge_labels)
        edge_accuracies.append(edge_accuracy)

        node_mse = float(F.mean_squared_error(output, target).array)
        node_mses.append(node_mse)

        if iterator.is_new_epoch:
            break

    put_log(iterator.epoch, np.mean(nll_valid), np.mean(kl_valid),
            np.mean(edge_accuracies), np.mean(node_mses), 'valid')

    chainer.config.train = True
    chainer.config.enable_backprop = True


def test(iterator, gpu, timesteps, encoder, decoder, rel_send, rel_rec, edge_types, temp, var):
    nll_test = []
    kl_test = []
    edge_accuracies = []
    node_mses = []

    chainer.config.train = False
    chainer.config.enable_backprop = False

    while True:
        inputs = iterator.next()
        node_features, edge_labels = dataset.concat_examples(inputs, device=gpu)

        data_encoder = node_features[:, :, :timesteps, :]
        data_decoder = node_features[:, :, -timesteps:, :]

        # logits: [batch_size, num_edges, edge_types]
        logits = encoder(data_encoder, rel_send, rel_rec)  # inverse func. of softmax
        edges = F.gumbel_softmax(logits, tau=temp, axis=2)
        edge_probs = F.softmax(logits, axis=2)
        # edges, edge_probs: [batch_size, num_edges, edge_types]

        # validation output uses teacher forcing
        output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

        target = data_decoder[:, :, 1:, :]
        num_nodes = node_features.shape[1]

        loss_nll = get_nll_gaussian(output, target, var)
        loss_kl = get_kl_categorical_uniform(edge_probs, num_nodes, edge_types)

        nll_test.append(float(loss_nll.array))
        kl_test.append(float(loss_kl.array))

        edge_accuracy = get_edge_accuracy(logits.array, edge_labels)
        edge_accuracies.append(edge_accuracy)

        node_mse = float(F.mean_squared_error(output, target).array)
        node_mses.append(node_mse)

        if iterator.is_new_epoch:
            break

    put_log(iterator.epoch, np.mean(nll_test), np.mean(kl_test),
            np.mean(edge_accuracies), np.mean(node_mses), 'test')

    chainer.config.train = True
    chainer.config.enable_backprop = True


def parse_log(out, split='train'):
    loss_nlls = []
    loss_kls = []
    edge_accs = []
    node_mses = []

    for line in open(os.path.join(out, 'log.txt')):
        if split not in line:
            continue
        ret = re.search('Epoch', line)
        if ret:
            loss_nll = float(re.search('loss_nll:\s([0-9\.-]+)', line).groups()[0])
            loss_kl = float(re.search('loss_kl:\s([0-9\.-]+)', line).groups()[0])
            edge_acc = float(re.search('edge_accuracy:([0-9\.-]+)', line).groups()[0])
            node_mse = float(re.search('node_mse:\s([0-9\.-]+)', line).groups()[0])

            loss_nlls.append(loss_nll)
            loss_kls.append(loss_kl)
            edge_accs.append(edge_acc)
            node_mses.append(node_mse)

    if len(loss_nlls) == 0:
        return

    fig, ax1 = plt.subplots()
    ax1.plot(loss_nlls, c='red')
    ax1.set_ylabel('loss_nll')
    ax2 = ax1.twinx()
    ax2.plot(loss_kls, c='blue')
    ax2.set_ylabel('loss_kl')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'loss_{}.png'.format(split)))
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.plot(edge_accs, c='red')
    ax1.set_ylabel('edge_accuracy')
    ax2 = ax1.twinx()
    ax2.plot(node_mses, c='blue')
    ax2.set_ylabel('node_mse')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'acc_{}.png'.format(split)))
    plt.close(fig)


if __name__ == '__main__':
    args = get_args()

    # Set a random seed
    np.random.seed(args.seed)
    if args.gpu >= 0:
        backends.cuda.cupy.random.seed(args.seed)
    random.seed(args.seed)

    train_iter = get_iterator(args.train_loc_npy, args.train_vel_npy, args.train_edges_npy)
    valid_iter = get_iterator(args.valid_loc_npy, args.valid_vel_npy, args.valid_edges_npy)
    test_iter = get_iterator(args.test_loc_npy, args.test_vel_npy, args.test_edges_npy)

    # Store the training dataset information
    args.train_num_episodes = train_iter.dataset.num_episodes
    args.train_num_nodes = train_iter.dataset.num_nodes
    args.train_timesteps = train_iter.dataset.timesteps
    args.train_num_features = train_iter.dataset.num_features

    # Save arguments into a JSON file
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'))

    # Create models
    encoder = get_encoder(
        args.encoder, args.timesteps, args.feature_dims, args.encoder_hidden, args.edge_types,
        args.encoder_dropout, args.factor)

    decoder = get_decoder(
        args.decoder, args.timesteps, args.feature_dims, args.decoder_hidden, args.edge_types,
        args.decoder_dropout, args.skip_first)

    if args.gpu >= 0:
        encoder = encoder.to_gpu(args.gpu)
        decoder = decoder.to_gpu(args.gpu)

    # Create optimizers
    enc_optim = optimizers.Adam(alpha=args.lr).setup(encoder)
    dec_optim = optimizers.Adam(alpha=args.lr).setup(decoder)

    # Lists of node IDs
    rel_send, rel_rec = get_sender_receiver(args.train_num_nodes)
    if args.gpu >= 0:
        rel_send = backends.cuda.to_gpu(rel_send, args.gpu)
        rel_rec = backends.cuda.to_gpu(rel_rec, args.gpu)

    # Set MemoryHook for benchmarking
    if args.benchmark:
        from nri.hooks.malloc_hook import MallocHook
        with MallocHook():
            train(train_iter, args.gpu, encoder, decoder, enc_optim, dec_optim, rel_send, rel_rec,
                  args.edge_types, args.temp, args.prediction_steps, args.var, args.out, args.benchmark, args.lr_decay,
                  args.gamma)

    # Normal training mode
    while train_iter.epoch < args.epochs:
        train(train_iter, args.gpu, encoder, decoder, enc_optim, dec_optim, rel_send, rel_rec,
              args.edge_types, args.temp, args.prediction_steps, args.var, args.out, args.benchmark, args.lr_decay,
              args.gamma)
        valid(valid_iter, args.gpu, encoder, decoder, rel_send, rel_rec, args.edge_types, args.temp, args.var)
        test(test_iter, args.gpu, args.timesteps, encoder, decoder, rel_send, rel_rec, args.edge_types, args.temp,
             args.var)

        parse_log(args.out, 'train')
        parse_log(args.out, 'valid')
        parse_log(args.out, 'test')
