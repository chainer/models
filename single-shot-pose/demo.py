import argparse
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainercv.utils import read_image
from chainercv.visualizations import vis_image

from lib.ssp import SSPYOLOv2
from lib.vis_point import vis_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('image')
    args = parser.parse_args()

    img = read_image(args.image)
    model = SSPYOLOv2()
    chainer.serializers.load_npz(args.pretrained_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    points, labels, scores = model.predict([img])
    point = points[0]
    label = labels[0]
    score = scores[0]

    vis_point(img, point[:1])
    plt.show()


if __name__ == '__main__':
    main()
