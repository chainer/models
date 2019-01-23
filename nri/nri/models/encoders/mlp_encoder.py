import logging

import chainer
import chainer.functions as F
import chainer.links as L

from nri.models import mlp


class MLPEncoder(chainer.Chain):

    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        # Xavier normal = Glorot normal
        w = chainer.initializers.GlorotNormal()

        self.factor = factor

        with self.init_scope():
            self.mlp1 = mlp.MLP(n_in, n_hid, n_hid, do_prob)
            self.mlp2 = mlp.MLP(n_hid * 2, n_hid, n_hid, do_prob)
            self.mlp3 = mlp.MLP(n_hid, n_hid, n_hid, do_prob)

            logger = logging.getLogger(__name__)
            if self.factor:
                self.mlp4 = mlp.MLP(n_hid * 3, n_hid, n_hid, do_prob)
                logger.info("Using factor graph MLP encoder.")
            else:
                self.mlp4 = mlp.MLP(n_hid * 2, n_hid, n_hid, do_prob)
                logger.info("Using MLP encoder.")
            self.fc_out = L.Linear(n_hid, n_out, initialW=w, initial_bias=0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # x: [batchsize, num_edges, feaure_dims]
        # rel_rec: [num_edges, num_nodes]
        # num_edges = num_nodes * (num_nodes - 1) (exclude self connection)
        incoming = F.matmul(rel_rec.T, x)
        # incoming: [batchsize, num_nodes, feature_dims]
        return incoming / incoming.shape[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # x: [batch_size, num_nodes, feature_dim]
        # rel_rec, rel_send: [num_edges, num_nodes]
        receivers = F.matmul(rel_rec, x)
        senders = F.matmul(rel_send, x)
        # receivers, senders: [batch_size, num_edges, feature_dim]
        edges = F.concat([receivers, senders], axis=2)  # along num_edges
        return edges

    def forward(self, inputs, rel_send, rel_rec):
        # Input shape: [batch_size, num_nodes, num_timesteps, num_dims]
        x = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        # New shape: [batch_size, num_nodes, num_timesteps * num_dims]

        # Obtain embeddings
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = F.concat((x, x_skip), axis=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = F.concat((x, x_skip), axis=2)  # Skip connection
            x = self.mlp4(x)

        # x shape: [batch_size, num_nodes * num_dims, n_hid]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # x shape: [batch_size * num_nodes * num_dims, n_hid]
        x = self.fc_out(x)
        # x shape: [batch_size * num_nodes * num_dims, n_out]
        x = x.reshape(inputs.shape[0], -1, x.shape[1])
        # x shape: [batch_size, num_nodes * num_dims, n_out]

        return x
