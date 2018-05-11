'''
ELMo usage example with character inputs.

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import chainer

from bilm import Batcher
from bilm import Elmo

# Location of pretrained LM and others.
vocab_file = 'vocab-2016-09-10.txt'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Build the Elmo with biLM and weight layers.
elmo = Elmo(
    options_file,
    weight_file,
    num_output_representations=1,
    requires_grad=False,
    do_layer_norm=False,
    dropout=0.)
# num_output_representations represents
# the number of weighted-sum patterns.
# that is, set 1 if using elmo at the input layer in another neural model.
#          set 2 if using elmo at both input and pre-output in a neural model.

# list of list of str. (i-th batch, j-th token, token's surface string)
# [1st_sentence = [1st word, 2nd word, ...],
#  2nd_sentence = [...]]
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create batches of data.
context_ids = batcher.batch_sentences(tokenized_context, add_bos_eos=False)
question_ids = batcher.batch_sentences(tokenized_question, add_bos_eos=False)
# numpy.ndarray or cupy.ndarray
# with shape (batchsize, max_length, max_character_length)
# default max_character_length = 50

# gpu id
# if you want to use cpu, set gpu=-1
# gpu = 0
gpu = -1
if gpu >= 0:
    # transfer the model to the gpu
    chainer.cuda.get_device_from_id(gpu).use()
    elmo.to_gpu()
    # transfer input data to the gpu
    context_ids = elmo.xp.asarray(context_ids)
    question_ids = elmo.xp.asarray(question_ids)

# Compute elmo outputs,
# i.e. weighted sum of multi-layer biLM's outputs.
context_embeddings = elmo.forward(context_ids)
question_embeddings = elmo.forward(question_ids)

"""
elmo's output is a dict with the following key-values:
    "elmo_representations": list of chainer.Variable.
        Each element has shape (batchsize, max_length, dim).
        i-th element represents weighted sum using the (i+1)-th weight pattern.
    "mask": numpy.ndarray with shape (batchsize, max_length).
        This mask represents the positions of padded fake values.
        The value of mask[j, k] represents
        if elmo_representations[j, k, :] is valid or not.
        For example, if 1st sentence has 9 tokens and 2nd one has 11,
        the mask is [[1 1 1 1 1 1 1 1 1 0 0]
                     [1 1 1 1 1 1 1 1 1 1 1]]
    "elmo_layers": list of chainer.Variable.
        Each element has shape (batchsize, max_length, dim).
        i-th element represents the output of i-th layer of biLM in elmo.
        Note 0th element is word embedding as input to biLM.
"""

print(len(context_embeddings['elmo_representations']),
      [x.shape for x in context_embeddings['elmo_representations']])
print(context_embeddings['elmo_representations'][0])
print(len(context_embeddings['elmo_layers']),
      [x.shape for x in context_embeddings['elmo_layers']])

print(type(context_embeddings['elmo_representations'][0]))
print(context_embeddings['elmo_representations'][0].shape)
# print(context_embeddings['elmo_representations'])
# print(context_embeddings['elmo_layers'][0])
# print(context_embeddings['elmo_layers'][1])
# print(context_embeddings['elmo_layers'][2])
