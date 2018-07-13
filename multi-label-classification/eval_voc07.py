import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.chainer_experimental.datasets.sliceable import\
    TransformDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.links import ResNet50
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from lib.eval_multi_label_classification import eval_multi_label_classification
from lib.transform import BatchTransform
from lib.transform import bbox_to_multi_label


class PredictFunc(object):

    def __init__(self, model, thresh=0):
        self.model = model
        self.thresh = thresh

    def __call__(self, imgs):
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            transform = BatchTransform(self.model.mean)
            imgs = transform(imgs)
            imgs = self.model.xp.array(imgs)
            scores = self.model(imgs)
            probs = chainer.cuda.to_cpu(F.sigmoid(scores).data)

        labels = []
        scores = []
        for prob in probs:
            label = np.where(prob >= self.thresh)[0]
            labels.append(label)
            scores.append(prob[label])
        return labels, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    args = parser.parse_args()

    model = ResNet50(
        pretrained_model=args.pretrained_model,
        n_class=len(voc_bbox_label_names), arch='he')
    model.pick = 'fc6'
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = VOCBboxDataset(
        split='test', year='2007', use_difficult=False)
    dataset = TransformDataset(
        dataset, ('img', 'bbox'), bbox_to_multi_label)
    iterator = iterators.SerialIterator(
        dataset, 8, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        PredictFunc(model, thresh=0), iterator,
        hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values
    pred_labels, pred_scores = out_values
    gt_labels, = rest_values

    result = eval_multi_label_classification(
        pred_labels, pred_scores, gt_labels)
    print()
    print('mAP: {:f}'.format(result['map']))
    for l, name in enumerate(voc_bbox_label_names):
        if result['ap'][l]:
            print('{:s}: {:f}'.format(name, result['ap'][l]))
        else:
            print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
