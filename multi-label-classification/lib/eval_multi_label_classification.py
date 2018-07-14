from __future__ import division

import numpy as np

from collections import defaultdict

from chainercv.evaluations import calc_detection_voc_ap


def eval_multi_label_classification(
        pred_labels, pred_scores, gt_labels):
    prec, rec = calc_multi_label_classification_prec_rec(
        pred_labels, pred_scores, gt_labels)
    ap = calc_detection_voc_ap(prec, rec)
    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_multi_label_classification_prec_rec(
        pred_labels, pred_scores, gt_labels):

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for pred_label, pred_score, gt_label in zip(
            pred_labels, pred_scores, gt_labels):
        indices = np.argsort(pred_label)
        pred_label = pred_label[indices]
        pred_score = pred_score[indices]
        np.testing.assert_equal(
            pred_label, np.arange(len(pred_label)))

        n_class = pred_label.shape[0]
        for lb in pred_label:
            if lb in list(gt_label):
                n_pos[lb] += 1
                match[lb].append(1)
            else:
                match[lb].append(0)
            score[lb].append(pred_score[lb])

    n_class = max(n_pos.keys()) + 1
    prec = [None] * n_class
    rec = [None] * n_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec
