from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser

import numpy as np

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('path', help='path to the pre-trained word vector file')
    ap.add_argument('--dim', default=300, type=int, help='dimension of the pre-trained word vector')
    args = ap.parse_args()

    glove_path = args.path
    embed_path = args.path.replace('.txt', '.npy')
    word_path = args.path.replace('.txt', '.word')
    print('Convert {} to {}'.format(glove_path, embed_path))

    N_lines = sum(1 for l in open(glove_path))
    embed_matrix = np.zeros((N_lines, args.dim), dtype='float32')
    with open(glove_path, 'r') as fi, open(word_path, 'w') as fo:
        for i, line in enumerate(fi):
            parts = line.rstrip().split(' ')
            assert len(parts) == (args.dim + 1)
            word = parts[0]
            vec = np.array(parts[1:], dtype='float32')
            embed_matrix[i] = vec
            fo.write(word + '\n')
    np.save(embed_path, embed_matrix)
