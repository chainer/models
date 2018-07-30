import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils import read_image


class LinemodDataset(GetterDataset):

    def __init__(self, base_dir, obj_name='ape', split='test'):
        super(LinemodDataset, self).__init__()
        split_path = os.path.join(
            base_dir, 'LINEMOD', obj_name, '{}.txt'.format(split))
        self.base_dir = base_dir
        with open(split_path, 'r') as f:
            self.img_paths = f.readlines()

        self.add_getter(('img', 'point', 'label'), self._get_example)

    def __len__(self):
        return len(self.img_paths)

    def _get_example(self, i):
        img_path = os.path.join(self.base_dir, self.img_paths[i].rstrip())
        img = read_image(img_path)

        anno_path = img_path.replace(
            'images', 'labels').replace(
                'JPEGImages', 'labels').replace(
                    '.jpg', '.txt').replace('.png', '.txt')

        anno = np.zeros(50*21)
        if os.path.getsize(anno_path):
            _, H, W = img.shape
            tmp = read_truths_args(anno_path, 8.0/W)
            size = tmp.size
            if size > 50*21:
                anno = tmp[0:50*21]
            elif size > 0:
                anno[0:size] = tmp
        anno = anno.reshape(-1, 21)
        anno = anno[:truths_length(anno)]
        point = anno[:, 1:19].reshape(-1, 9, 2).astype(np.float32)
        point[:, :, 0] *= W
        point[:, :, 1] *= H
        label = anno[:, 0].astype(np.int32)
        return img, point, label


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i


def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        truths = truths.reshape(truths.size//21, 21)
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        new_truths.append(
            [truths[i][0], truths[i][1], truths[i][2],
             truths[i][3], truths[i][4], truths[i][5],
             truths[i][6], truths[i][7], truths[i][8],
             truths[i][9], truths[i][10], truths[i][11],
             truths[i][12], truths[i][13], truths[i][14],
             truths[i][15], truths[i][16], truths[i][17],
             truths[i][18]])
    return np.array(new_truths)


if __name__ == '__main__':
    dataset = LinemodDataset('../')
    from vis_point import vis_point
    import matplotlib.pyplot as plt
    img, point, label = dataset[0]
    vis_point(img, point)
    plt.show()
