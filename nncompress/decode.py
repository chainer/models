# -*- coding: utf-8 -*-
import argparse
import os

import chainer
import numpy as np
from chainer.dataset.convert import concat_examples
from chainer.iterators import SerialIterator

from compressor.net import EmbeddingCompressor
from compressor.resource import Resource


def main():
    parser = argparse.ArgumentParser(description='Embedding Compressor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of Instances Contained in Each MiniBatch')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (Negative Value Indicates CPU)')

    # Arguments for the dataset / vocabulary path
    parser.add_argument('--model', dest='model', required=True, type=os.path.abspath, help='Model file from train.py')
    parser.add_argument('--vocab', dest='vocab_file', required=True, type=os.path.abspath,
                        help='Vocabulary file (.word)')
    parser.add_argument('--embed', dest='embed_matrix', required=True, type=os.path.abspath,
                        help='Original Embedding matrix (.npy)')
    args = parser.parse_args()

    resource = Resource(args, train=False)
    logger = resource.logger

    resource.load_config()
    model_path = resource.args.model

    vocabs = [l.strip() for l in open(args.vocab_file, 'r')]
    test_data = list(range(len(vocabs)))
    test_iter = SerialIterator(dataset=test_data, batch_size=args.batchsize, shuffle=False, repeat=False)
    embed_mat = np.load(args.embed_matrix)
    assert embed_mat.shape[0] == len(vocabs)
    model = EmbeddingCompressor(
        n_codebooks=resource.config['n_codebooks'],
        n_centroids=resource.config['n_centroids'],
        n_vocab=len(vocabs),
        embed_dim=embed_mat.shape[1],
        tau=resource.config['tau'],
        embed_mat=embed_mat
    )

    logger.info('Loading the model from [{}]'.format(os.path.abspath(model_path)))
    chainer.serializers.load_npz(model_path, model, strict=False)

    if args.gpu >= 0:
        logger.info('Sending model to GPU #{}'.format(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    # Export Codebook Params
    codebook_dest = os.path.join(resource.output_dir, '{}.codebook.npy'.format(resource.model_name))
    logger.info('Exporting Codebook: {}'.format(codebook_dest))
    model.export(codebook_dest)
    logger.info('Codebook Exported!')

    # Export Codebook Content
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        dest = os.path.join(resource.output_dir, '{}.codes'.format(resource.model_name))
        logger.info('Codes Output: {}'.format(dest))
        with open(dest, 'w') as fo:
            for batch in test_iter:
                conv_batch = concat_examples(batch, device=args.gpu)
                target_v = [vocabs[x] for x in batch]
                for out in model.retrieve_codes(conv_batch, target_v):
                    fo.write(out + '\n')
                    logger.info(out)
    logger.info('Done!'.format(dest))


if __name__ == "__main__":
    main()
