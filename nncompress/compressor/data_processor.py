# -*- coding: utf-8 -*-

import os

import logzero
import numpy as np
from logzero import logger

DATA_TYPE = ('train', 'dev', 'test')


class DataProcessor(object):
    def __init__(self, log_name):
        self.embed_matrix = None

        self.logger = logger
        logzero.logfile(log_name)

    def load_embed_matrix(self, path):
        self.logger.info('Loading Embedding Matrix from {}'.format(os.path.abspath(path)))
        self.embed_matrix = np.load(path)
        n_vocab, embed_dim = self.embed_matrix.shape
        self.logger.info(
            'Embedding Matrix Shape: {} (Vocab = {}, Dimension = {})'.format(self.embed_matrix.shape, n_vocab,
                                                                             embed_dim))

    def load_data(self, data_type):
        assert data_type in DATA_TYPE
        assert self.embed_matrix is not None

        n_vocab, embed_dim = self.embed_matrix.shape
        if data_type == 'train':
            dataset = list(range(n_vocab))
        elif data_type == 'dev':
            dataset = list(range(n_vocab))
            np.random.shuffle(dataset)
            dataset = dataset[:2000]  # dev size: 2000
        self.logger.info('{} data is successfully loaded'.format(data_type))
        self.logger.info('{} data contains {} instances'.format(data_type, len(dataset)))
        return dataset
