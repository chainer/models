import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import ResNet50
from chainercv import utils
from chainercv.visualizations import vis_image

from eval_voc07 import PredictFunc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = ResNet50(
        pretrained_model=args.pretrained_model,
        n_class=len(voc_bbox_label_names))
    model.pick = 'fc6'
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    predict_func = PredictFunc(model, thresh=0.5)
    labels, scores = predict_func([img])
    label = labels[0]
    score = scores[0]

    print('predicted labels')
    for lb, sc in zip(label, score):
        print('names={}  score={:.4f}'.format(
            voc_bbox_label_names[lb], sc))

    vis_image(img)
    plt.show()


if __name__ == '__main__':
    main()
