from chainer import reporter
from chainer.backends import cuda
from chainer.training.extensions import Evaluator


class CopyTransformerEvaluationFunction:

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def __call__(self, **kwargs):
        data = kwargs.pop('data')
        labels = kwargs.pop('label')

        with cuda.Device(self.device):
            data = self.net.xp.array(data)
            labels = self.net.xp.array(labels)

            prediction = self.net.predict(data)
            # part accuracy is the accuracy for each number and accuracy is the accuracy
            # for the complete vector of numbers
            part_accuracy, accuracy = self.calc_accuracy(prediction, labels)

        reporter.report({
            "part_accuracy": part_accuracy,
            "accuracy": accuracy
        })

    def calc_accuracy(self, predictions, labels):
        correct_lines = 0
        correct_parts = 0
        for predicted_item, item in zip(predictions, labels):
            # count the number of correct numbers
            accuracy_result = (predicted_item == item).sum()
            correct_parts += accuracy_result

            # if all numbers are correct, we can also increase the number of correct lines/vectors
            if accuracy_result == predictions.shape[1]:
                correct_lines += 1

        return correct_parts / predictions.size, correct_lines / len(predictions)


class CopyTransformerEvaluator(Evaluator):

    def evaluate(self):
        summary = reporter.DictSummary()
        eval_func = self.eval_func or self._targets['main']

        observation = {}
        with reporter.report_scope(observation):
            # we always use the same array for testing, since this is only an example ;)
            data = eval_func.net.xp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype='int32')
            eval_func(data=data, label=data)

        summary.add(observation)
        return summary.compute_mean()
