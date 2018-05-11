import chainer
from bilm import Batcher
from bilm import Elmo
vocab_file = 'vocab-2016-09-10.txt'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

batcher = Batcher(vocab_file, 50)
elmo = Elmo(
    options_file,
    weight_file,
    num_output_representations=1,
    requires_grad=False,
    do_layer_norm=False,
    dropout=0.)

raw_sents = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_sents = [sentence.split() for sentence in raw_sents]
batched_ids = batcher.batch_sentences(tokenized_sents, add_bos_eos=False)
embeddings = elmo.forward(batched_ids)

print(type(embeddings['elmo_representations'][0]))
# <class 'chainer.variable.Variable'>
print(embeddings['elmo_representations'][0].shape)
# (2, 11, 1024) = (batchsize, max_sentence_length, dim)
