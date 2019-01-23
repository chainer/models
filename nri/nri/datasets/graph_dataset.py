import logging
import os

import chainer
import numpy as np


class GraphDataset(chainer.dataset.DatasetMixin):

    def __init__(self, loc_npy, vel_npy, edges_npy):
        self.features, self.edges = self.prepare_datasets(loc_npy, vel_npy, edges_npy)

        self.num_episodes = self.features.shape[0]
        self.num_nodes = self.features.shape[1]
        self.timesteps = self.features.shape[2]
        self.num_features = self.features.shape[3]

        logger = logging.getLogger(__name__)
        logger.info('num_episodes: {}'.format(self.num_episodes))
        logger.info('num_nodes: {}'.format(self.num_nodes))
        logger.info('timesteps: {}'.format(self.timesteps))
        logger.info('num_features: {}'.format(self.num_features))

    def __len__(self):
        return len(self.features)

    def prepare_datasets(self, loc_npy, vel_npy, edges_npy):
        """ Load data from .npy files.

        The shape of loc: (num_samples, num_timesteps, 2, num_nodes)
            "2" means that a location is represented in a 2D coordinate.

        The shape of vel: (num_samples, num_timesteps, 2, num_nodes)
            "2" means that a velocity is represented in a 2D vector.

        The shape of edge: (num_samples, num_nodes, num_nodes)
            It has a connectivity matrix of the nodes.

        """
        loc = np.load(loc_npy)
        vel = np.load(vel_npy)
        edges = np.load(edges_npy)

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_nodes = loc.shape[3]

        loc_max = loc.max()
        loc_min = loc.min()
        vel_max = vel.max()
        vel_min = vel.min()

        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1
        vel = (vel - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc = np.transpose(loc, [0, 3, 1, 2])
        vel = np.transpose(vel, [0, 3, 1, 2])
        features = np.concatenate([loc, vel], axis=3)

        edges = np.reshape(edges, [-1, num_nodes ** 2])
        edges = np.array((edges + 1) / 2)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges = edges[:, off_diag_idx]

        return features.astype(np.float32), edges.astype(np.int32)


    def get_example(self, i):
        return self.features[i], self.edges[i]
