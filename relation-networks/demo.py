import argparse

import chainer
from PIL import Image
import numpy as np

import dataset
from model import RelationNetwork


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str,
                        default='result/snapshot_10000',
                        help='Model snapshot file path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--sort-of-clevr-path', type=str,
                        default='sort_of_clevr.pkl',
                        help='Sort-of-CLEVR pickle file')
    parser.add_argument('--sort-of-clevr-index', type=int,
                        default=0,
                        help='Sort-of-CLEVR index')
    parser.add_argument('--out', type=str, default='result/demo.png',
                        help='Output directory')
    args = parser.parse_args()

    dataset, clevr = dataset.get_sort_of_clevr(args.sort_of_clevr_path)
    img, question, answer_token = dataset[args.sort_of_clevr_index]

    relation_network = RelationNetwork(len(clevr.vocab))
    chainer.serializers.load_npz(args.snapshot, relation_network)

    if args.gpu >= 0:
        cuda = chainer.backends.cuda
        gpu = args.gpu

        cuda.get_device_from_id(gpu).use()

        relation_network.to_gpu()
        img = cuda.to_gpu(img)
        question = cuda.to_gpu(question)

    logits = relation_network(img[None], question[None])  # Add batch axis.
    pred = relation_network.xp.argmax(logits.array)

    if args.gpu >= 0:
        img = img.get()
        question = question.get()
        pred = pred.get()

    question_decoded = clevr.decode_question(question)
    pred_decoded = clevr.decode_answer(pred)
    answer_decoded = clevr.decode_answer(answer_token)
    print('Question\tPrediction\tAnswer')
    print(question_decoded, '\t', pred_decoded, '\t', answer_decoded)

    img = Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0), clevr.mode)
    img.save(args.out)
    print('Image was saved to', args.out)
