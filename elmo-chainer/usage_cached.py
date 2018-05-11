'''
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''

import h5py

from bilm import dump_bilm_embeddings

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the dataset file.
dataset_file = 'dataset_file.txt'
with open(dataset_file, 'w') as fout:
    for sentence in tokenized_context + tokenized_question:
        fout.write(' '.join(sentence) + '\n')


# Location of pretrained LM.
vocab_file = 'vocab-2016-09-10.txt'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

# Dump the embeddings to a file. Run this once for your dataset.
embedding_file = 'elmo_embeddings.hdf5'

# gpu id
# if you want to use cpu, set gpu=-1
gpu = -1
# batchsize
# encoding each token is inefficient
# encoding too many tokens is difficult due to memory
batchsize = 32

dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file,
    gpu=gpu, batchsize=batchsize
)

# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(embedding_file, 'r') as fin:
    second_sentence_embeddings = fin['1'][...]
    print(second_sentence_embeddings.shape)
    # (n_layers=3, sequence_length, embedding_dim)
    print(second_sentence_embeddings)
