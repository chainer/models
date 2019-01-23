#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re

import chainer
import chainer.functions as F
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from chainer import backends
from chainer import dataset
from chainer import iterators
from chainer import serializers

from nri.datasets import graph_dataset
from nri.models import decoders
from nri.models import encoders


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


def get_sender_receiver(num_nodes):
    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)

    # Split receiver and sender node IDs
    rel_send, rel_rec = np.where(off_diag)

    # Make them 1-hot vectors
    rel_send = np.eye(len(np.unique(rel_send)))[rel_send]
    rel_rec = np.eye(len(np.unique(rel_rec)))[rel_rec]

    return rel_send.astype(np.float32), rel_rec.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--args-file', type=str)
    parser.add_argument('--encoder-snapshot', type=str)
    parser.add_argument('--decoder-snapshot', type=str)
    args = parser.parse_args()

    train_args = json.load(open(args.args_file))
    print(json.dumps(train_args, indent=4))

    encoder = get_encoder(
        train_args['encoder'], train_args['timesteps'], train_args['feature_dims'],
        train_args['encoder_hidden'], train_args['edge_types'], train_args['encoder_dropout'],
        not train_args['no_factor'])

    decoder = get_decoder(
        train_args['decoder'], train_args['timesteps'], train_args['feature_dims'],
        train_args['decoder_hidden'], train_args['edge_types'], train_args['decoder_dropout'],
        train_args['skip_first'])

    serializers.load_npz(args.encoder_snapshot, encoder)
    serializers.load_npz(args.decoder_snapshot, decoder)

    if args.gpu >= 0:
        encoder.to_gpu(args.gpu)
        decoder.to_gpu(args.gpu)

    # Lists of node IDs
    rel_send, rel_rec = get_sender_receiver(train_args['train_num_nodes'])
    if args.gpu >= 0:
        rel_send = backends.cuda.to_gpu(rel_send, args.gpu)
        rel_rec = backends.cuda.to_gpu(rel_rec, args.gpu)

    ds = graph_dataset.GraphDataset(
        train_args['test_loc_npy'],
        train_args['test_vel_npy'],
        train_args['test_edges_npy'])
    test_iter = iterators.SerialIterator(ds, 1)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    for i in range(5):
        inputs = test_iter.next()
        node_features, edge_labels = dataset.concat_examples(inputs, device=args.gpu)

        data_encoder = node_features[:, :, :train_args['timesteps'], :]
        data_decoder = node_features[:, :, train_args['timesteps']:, :]

        # logits: [batch_size, num_edges, edge_types]
        logits = encoder(data_encoder, rel_send, rel_rec)  # inverse func. of softmax
        edges = F.gumbel_softmax(logits, tau=train_args['temp'], axis=2)  # edge sampling
        edge_probs = F.softmax(logits, axis=2)
        # edges, edge_probs: [batch_size, num_edges, edge_types]

        # validation output uses teacher forcing
        output = decoder(data_decoder, edges, rel_rec, rel_send, data_decoder.shape[2])

        fig = plt.figure()
        plt.tight_layout()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('transparent: given, solid: prediction, dashed: ground-truth')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        frames = []

        data_encoder = backends.cuda.to_cpu(data_encoder)
        for t in range(train_args['timesteps']):
            frame = []
            for node_i in range(train_args['train_num_nodes']):
                node = plt.plot(
                    data_encoder[0, node_i, :t, 0], data_encoder[0, node_i, :t, 1], '-',
                    c=colors[node_i], alpha=0.5)
                frame.extend(node)
            frames.append(frame)

        output.to_cpu()
        output = output.array
        data_decoder = backends.cuda.to_cpu(data_decoder)

        for t in range(output.shape[2]):
            frame = []
            for node_i in range(train_args['train_num_nodes']):
                given = plt.plot(
                    data_encoder[0, node_i, :, 0], data_encoder[0, node_i, :, 1], '-',
                    c=colors[node_i], alpha=0.5)
                frame.extend(given)

            for node_i in range(train_args['train_num_nodes']):
                node = plt.plot(output[0, node_i, :t, 0], output[0, node_i, :t, 1], '-', c=colors[node_i])
                frame.extend(node)

            for node_i in range(train_args['train_num_nodes']):
                node = plt.plot(data_decoder[0, node_i, :t, 0], data_decoder[0, node_i, :t, 1], '--', c=colors[node_i])
                frame.extend(node)
            frames.append(frame)

        a = animation.ArtistAnimation(fig, frames, interval=100)
        a.save('images/result_{}.gif'.format(i), 'imagemagick')
        # a.save('images/result_{}.mp4'.format(i), 'ffmpeg')
