import numpy as np

import chainer
import chainer.functions as F
from chainer import reporter

from .transform import BatchTransform


def calc_accuracy(pred_scores, gt_labels):
    # https://arxiv.org/pdf/1612.03663.pdf
    # end of section 3.1
    pred_probs = F.sigmoid(pred_scores).data

    accs = []
    n_pos = []
    n_pred = []
    for pred_prob, gt_label in zip(pred_probs, gt_labels):
        gt_label = chainer.cuda.to_cpu(gt_label)
        pred_prob = chainer.cuda.to_cpu(pred_prob)
        pred_label = np.where(pred_prob > 0.5)[0]

        correct = np.intersect1d(gt_label, pred_label)
        diff_gt = np.setdiff1d(gt_label, correct)
        diff_pred = np.setdiff1d(pred_label, correct)
        accs.append(
            len(correct) / (len(correct) + len(diff_gt) + len(diff_pred)))
        n_pos.append(len(gt_label))
        n_pred.append(len(pred_label))
    return {
        'accuracy': np.mean(accs),
        'n_pos': np.mean(n_pos),
        'n_pred': np.mean(n_pred)}


class MultiLabelClassifier(chainer.Chain):

    def __init__(self, model, loss_scale):
        super(MultiLabelClassifier, self).__init__()
        with self.init_scope():
            self.model = model
        self.loss_scale = loss_scale

    def __call__(self, x, labels):
        x = BatchTransform(self.model.mean)(x)
        x = self.xp.array(x)
        scores = self.model(x)

        B, n_class = scores.shape[:2]
        one_hot_labels = self.xp.zeros((B, n_class), dtype=np.int32)
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        # sigmoid_cross_entropy normalizes the loss
        # by the size of batch and the number of classes.
        # It works better to remove the normalization factor
        # of the number of classes.
        loss = self.loss_scale * F.sigmoid_cross_entropy(
            scores, one_hot_labels)

        result = calc_accuracy(scores, labels)
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': result['accuracy']}, self)
        reporter.report({'n_pred': result['n_pred']}, self)
        reporter.report({'n_pos': result['n_pos']}, self)
        return loss
