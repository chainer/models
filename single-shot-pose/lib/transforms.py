import numpy as np

from chainercv.datasets import VOCBboxDataset

from .resize import resize


def random_crop(img, point, jitter=0.2):
    _, H, W = img.shape
    dw = int(W * jitter)
    dh = int(H * jitter)

    pleft = np.random.randint(-dw, dw)
    pright = np.random.randint(-dw, dw)
    ptop = np.random.randint(-dh, dh)
    pbot = np.random.randint(-dh, dh)

    out_width = W - pleft - pright
    out_height = H - ptop - pbot

    ptop = max((ptop, 0))
    pleft = max((pleft, 0))
    img = img[:, ptop:ptop + out_height, pleft:pleft + out_width]

    point = point.copy()
    point[:, :, 0] -= pleft
    point[:, :, 1] -= ptop
    return img, point


class Transform(object):

    def __init__(self, random=None):
        self.voc = VOCBboxDataset(
            year='2012', split='trainval').slice[:, 'img']
        self.random = random

    def __call__(self, in_data):
        if self.random is None:
            # Different seeds on different processes
            self.random = np.random.RandomState()
        fg_img, point, label, msk = in_data
        _, H, W = fg_img.shape

        index = self.random.randint(0, len(self.voc))
        img = self.voc[index]
        img = resize(img, (H, W))
        img[:, msk] = fg_img[:, msk]

        img, point = random_crop(img, point)

        # skipping color related augmentation
        return img, point, label


if __name__ == '__main__':
    from linemod_dataset import LinemodDataset
    from vis_point import vis_point
    import matplotlib.pyplot as plt
    from chainercv.chainer_experimental.datasets.sliceable import \
        TransformDataset

    dataset = LinemodDataset('..', split='train', return_msk=True)
    dataset = TransformDataset(dataset, ('img', 'point', 'label'), Transform())
    img, point, label = dataset[0]
    vis_point(img, point)
    plt.show()
