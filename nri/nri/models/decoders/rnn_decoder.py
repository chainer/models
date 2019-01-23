import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class RNNDecoder(chainer.Chain):

    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()

        self.msg_fc1 = chainer.ChainList(
            [L.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = chainer.ChainList(
            [L.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = L.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = L.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = L.Linear(n_hid, n_hid, bias=False)

        self.input_r = L.Linear(n_in_node, n_hid, bias=True)
        self.input_i = L.Linear(n_in_node, n_hid, bias=True)
        self.input_n = L.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = L.Linear(n_hid, n_hid)
        self.out_fc2 = L.Linear(n_hid, n_hid)
        self.out_fc3 = L.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = F.matmul(rel_rec, hidden)
        senders = F.matmul(rel_send, hidden)
        pre_msg = F.concat([receivers, senders], axis=-1)

        all_msgs = np.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()