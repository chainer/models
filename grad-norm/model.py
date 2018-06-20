import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


def square_loss(ys, ts):
    # return F.mean(F.sqrt((ys - ts) ** 2 + 1e-5), axis=(0, 2))
    return F.mean((ys - ts) ** 2 + 1e-5, axis=(0, 2))


class RegressionTrainChain(chainer.Chain):

    def __init__(self, model):
        super(RegressionTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.weight = chainer.Parameter(
                np.ones(model.n_task, dtype=np.float32))

    def __call__(self, x, ts):
        B, n_task = ts.shape[:2]
        ys = self.model(x)

        task_loss = square_loss(ys, ts)
        return task_loss


class RegressionChain(chainer.Chain):

    def __init__(self, n_task):
        super(RegressionChain, self).__init__()
        self.n_task = n_task

        with self.init_scope():
            self.l1 = L.Linear(250, 100)
            self.l2 = L.Linear(100, 100)
            self.l3 = L.Linear(100, 100)
            self.l4 = L.Linear(100, 100)

            for i in range(self.n_task):
                setattr(self, 'task_{}'.format(i),
                        L.Linear(100, 100))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        outs = []
        for i in range(self.n_task):
            l = getattr(self, 'task_{}'.format(i))
            outs.append(l(h))
        return F.stack(outs, axis=1)

