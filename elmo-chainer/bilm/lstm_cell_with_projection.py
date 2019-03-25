"""
An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.
"""

# from typing import Optional, Tuple, List

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda


def get_dropout_mask(dropout_probability, tensor_for_masking):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    # Scale mask by 1/keep_prob to preserve output statistics.
    xp = cuda.get_array_module(tensor_for_masking)
    binary_mask = (xp.rand(tensor_for_masking.shape) > dropout_probability)
    dropout_mask = binary_mask / (1.0 - dropout_probability)
    return dropout_mask


class LstmCellWithProjection(chainer.Chain):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.

    Parameters
    ----------
    input_size : ``int``, required.
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required.
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell used for the LSTM.
    go_forward: ``bool``, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.

    Returns
    -------
    output_accumulator : ``torch.FloatTensor``
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 cell_size,
                 go_forward=True,
                 recurrent_dropout_probability=0.0,
                 memory_cell_clip_value=None,
                 state_projection_clip_value=None):
        super(LstmCellWithProjection, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = float(state_projection_clip_value)
        self.memory_cell_clip_value = float(memory_cell_clip_value)
        self.recurrent_dropout_probability = recurrent_dropout_probability

        with self.init_scope():
            # We do the projections for all the gates all at once.
            self.input_linearity = L.Linear(
                input_size, 4 * cell_size, nobias=True)
            self.state_linearity = L.Linear(
                hidden_size, 4 * cell_size, nobias=False)

            # Additional projection matrix for making the hidden state smaller.
            self.state_projection = L.Linear(
                cell_size, hidden_size, nobias=True)
            self.reset_parameters()

    def reset_parameters(self):
        # TODO: this is required when training
        """
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)
        """
        pass

    def forward(self,
                inputs,
                batch_lengths,
                initial_state=None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).

        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.shape[0]
        total_timesteps = inputs.shape[1]

        output_accumulator = self.xp.zeros(
            (batch_size, total_timesteps, self.hidden_size), 'f')
        if initial_state is None:
            full_batch_previous_memory = chainer.Variable(
                self.xp.zeros((batch_size, self.cell_size), 'f'))
            full_batch_previous_state = chainer.Variable(
                self.xp.zeros((batch_size, self.hidden_size), 'f'))
        else:
            # first dimension is just (layer * (1 + is_bidirection)), i.e., 1.
            full_batch_previous_state = F.squeeze(initial_state[0], axis=0)
            full_batch_previous_memory = F.squeeze(initial_state[1], axis=0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and \
           (self.training or chainer.confing.train):
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                        batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = full_batch_previous_memory[0: current_length_index + 1]
            # Shape (batch_size, hidden_size)
            previous_state = full_batch_previous_state[0: current_length_index + 1]
            # Shape (batch_size, input_size)
            timestep_input = inputs[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            # TODO: split_axis
            # TODO: cuda kernel
            input_gate = F.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                   projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = F.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                    projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = F.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                 projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = F.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                    projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.

            if self.memory_cell_clip_value:
                memory = F.clip(memory, -self.memory_cell_clip_value,
                                self.memory_cell_clip_value)

            # shape (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * F.tanh(memory)
            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(
                pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = F.clip(timestep_output,
                                         -self.state_projection_clip_value,
                                         self.state_projection_clip_value)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:
                timestep_output = timestep_output * \
                    dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = F.concat(
                [memory, full_batch_previous_memory[current_length_index + 1:]], axis=0)
            full_batch_previous_state = F.concat(
                [timestep_output, full_batch_previous_state[current_length_index + 1:]], axis=0)
            output_accumulator[0:current_length_index + 1, index] \
                = timestep_output.array

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (F.expand_dims(full_batch_previous_state, 0),
                       F.expand_dims(full_batch_previous_memory, 0))
        return output_accumulator, final_state
