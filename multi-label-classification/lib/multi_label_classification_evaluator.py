import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.utils import apply_to_iterator

from .eval_multi_label_classification import eval_multi_label_classification


class MultiLabelClassificationEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, predict_func, label_names=None):
        super(MultiLabelClassificationEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names
        self.predict_func = predict_func

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            self.predict_func, it)
        # delete unused iterators explicitly
        del in_values

        pred_labels, pred_scores = out_values
        gt_labels, = rest_values

        result = eval_multi_label_classification(
            pred_labels, pred_scores, gt_labels)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
