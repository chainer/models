import copy

from chainer import reporter
import chainer.training.extensions

from chainercv.utils import apply_to_iterator

from .eval_projected_3d_bbox import eval_projected_3d_bbox_single


class Projected3dBboxEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, vertex, intrinsics, diam):
        super(Projected3dBboxEvaluator, self).__init__(
            iterator, target)
        self.vertex = vertex
        self.intrinsics = intrinsics
        self.diam = diam

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)
        # delete unused iterators explicitly
        del in_values

        points, labels, scores = out_values
        gt_points, gt_labels = rest_values

        result = eval_projected_3d_bbox_single(
            points, scores, gt_points,
            self.vertex, self.intrinsics, diam=self.diam)
        report = result

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
