import chainer
import numpy as np


class RegressionDataset(chainer.dataset.DatasetMixin):

    def __init__(self, sigmas, epsilons):
        self.B = np.random.normal(scale=10, size=(100, 250)).astype(np.float32)
        assert epsilons.shape == (len(sigmas), 100, 250)
        self.sigmas=  np.array(sigmas).astype(np.float32)
        self.epsilons = np.array(epsilons).astype(np.float32)

    def __len__(self):
        return 100

    def get_example(self, i):
        # x = np.random.normal(scale=1/100, size=(250,)).astype(np.float32)
        x = np.random.uniform(-1, 1, size=(250,)).astype(np.float32)
        x /= np.linalg.norm(x)
        
        ys = []
        for i in range(len(self.sigmas)):
            ys.append(
                self.sigmas[i] * np.tanh((self.B + self.epsilons[i]).dot(x)))
        return x, np.stack(ys)
