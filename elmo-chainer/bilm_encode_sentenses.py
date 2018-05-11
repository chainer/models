'''
Encode dataset as biLM embeddings to a file.
'''
import argparse
import json

from bilm import dump_bilm_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--batchsize', '-b', type=int, default=32,
                    help='Minibatch size of computation')
parser.add_argument('--input', '-in', '-i', required=True,
                    help='Path of input text file')
parser.add_argument('--output', '-out', '-o', required=True,
                    help='Path of output file to be written')
args = parser.parse_args()
print(json.dumps(args.__dict__, indent=2))

# Location of pretrained LM.
vocab_file = 'vocab-2016-09-10.txt'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

dataset_file = args.input
embedding_file = args.output
assert args.input != args.output

dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file,
    gpu=args.gpu, batchsize=args.batchsize
)
