import logging

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import backends


class MLPDecoder(chainer.Chain):

    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid, do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()

        w = chainer.initializers.LeCunUniform(scale=(1. / np.sqrt(3)))
        b = self._bias_initializer

        with self.init_scope():
            self.msg_fc1 = chainer.ChainList(
                *[L.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
            self.msg_fc2 = chainer.ChainList(
                *[L.Linear(msg_hid, msg_out) for _ in range(edge_types)])
            self.out_fc1 = L.Linear(n_in_node + msg_out, n_hid, initialW=w, initial_bias=b)
            self.out_fc2 = L.Linear(n_hid, n_hid, initialW=w, initial_bias=b)
            self.out_fc3 = L.Linear(n_hid, n_in_node, initialW=w, initial_bias=b)

        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        logger = logging.getLogger(__name__)
        logger.info('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def _bias_initializer(self, array):
        scale = np.sqrt(1. / array.shape[0])
        return chainer.initializers.Uniform(scale)(array)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send, single_timestep_rel_type):
        # single_timestep_inputs: [batch_size, num_sequences, num_nodes, feature_dims]
        # single_timestep_rel_type: [batch_size, num_sequences, num_edges, edge_types]
        batch_size, num_sequences, num_edges, _ = single_timestep_rel_type.shape
        _, num_nodes = rel_rec.shape

        # Node2edge
        # rel_rec: [num_edges, num_nodes]
        # rel_send: [num_edges, num_nodes]
        receivers = F.matmul(rel_rec, single_timestep_inputs)
        senders = F.matmul(rel_send, single_timestep_inputs)
        pre_msg = F.concat([receivers, senders], axis=-1)
        # pre_msg: [batch_size, num_sequences, num_edges, 2 * feature_dims]
        pre_msg = F.reshape(pre_msg, [batch_size * num_sequences * num_edges, -1])

        all_msgs = chainer.Variable(
            pre_msg.xp.zeros((batch_size, num_sequences, num_edges, self.msg_out_shape),
                             dtype=single_timestep_rel_type.dtype))
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            # msg: [batch_size * num_sequences * num_edges, msg_hid]
            msg = F.reshape(msg, [batch_size, num_sequences, num_edges, -1])
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        # all_msgs: [batch_size, num_sequences, num_edges, msg_out_shape]
        # rel_rec: [num_edges, num_nodes]
        agg_msgs = F.matmul(rel_rec.T, all_msgs)

        # Skip connection
        aug_inputs = F.concat([single_timestep_inputs, agg_msgs], axis=-1)
        # aug_inputs: [batch_size, num_sequences, num_nodes, msg_out_shape + feature_dims]
        aug_inputs = F.reshape(aug_inputs, [batch_size * num_sequences * num_nodes, -1])

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), self.dropout_prob)
        pred = self.out_fc3(pred)
        pred = F.reshape(pred, [batch_size, num_sequences, num_nodes, -1])

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        # inputs: [batch_size, num_nodes, timesteps, feature_dim]
        inputs = inputs.transpose(0, 2, 1, 3)
        batch_size, timesteps = inputs.shape[:2]
        # inputs: [batch_size, timesteps, num_nodes, feature_dim]

        # rel_type: [batch_size, num_edges, edge_types]
        _, num_edges, edge_types = rel_type.shape
        # Repeat rel_type "timesteps" times
        rel_type = F.broadcast_to(
            rel_type[:, None, :, :], [batch_size, timesteps, num_edges, edge_types])

        assert (pred_steps <= timesteps)

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        _, num_sub_sequences, _, _ = last_pred.shape
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        preds = [[] for _ in range(num_sub_sequences)]

        # Run n prediction steps
        for _ in range(pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, curr_rel_type)
            for seq_i in range(num_sub_sequences):
                preds[seq_i].append(last_pred[:, seq_i:seq_i + 1, :, :])

        for seq_i in range(num_sub_sequences):
            preds[seq_i] = F.concat(preds[seq_i], axis=1)
        preds = F.concat(preds, axis=1)
        pred_all = preds[:, :(inputs.shape[1] - 1), :, :]

        return pred_all.transpose(0, 2, 1, 3)
