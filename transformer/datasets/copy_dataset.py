import copy
import numpy
from chainer.dataset import DatasetMixin


class CopyDataset(DatasetMixin):

    def __init__(self, max_value=1000, max_len=10, start_symbol=1):
        super().__init__()
        self.max_value = max_value
        self.max_len = max_len
        self.start_symbol = start_symbol

    def __len__(self):
        return 10000

    def get_example(self, i):
        # we randomly create a vector of `max_len` numbers and set the first element to be the start symbol
        data = numpy.random.randint(1, self.max_value, size=self.max_len, dtype=numpy.int32)
        data[0] = self.start_symbol
        return {
            "data": data,
            "label": copy.deepcopy(data),
        }
