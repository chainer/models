import numpy as np
import os
import warnings


from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.sbd import sbd_utils
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

try:
    import scipy
    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn(
            'SciPy is not installed in your environment,'
            'so the dataset cannot be loaded.'
            'Please install SciPy to load dataset.\n\n'
            '$ pip install scipy')


class SBDBboxDataset(GetterDataset):

    def __init__(self, data_dir='auto'):
        super(SBDBboxDataset, self).__init__()

        _check_available()

        file_dir = os.path.dirname(os.path.realpath(__file__))
        f = open(os.path.join(file_dir, 'splits/sbd_trainaug.txt'))
        self.ids = []
        for l in f.readlines():
            path = l.split()[0]
            self.ids.append(os.path.split(os.path.splitext(path)[0])[1])

        if data_dir == 'auto':
            data_dir = sbd_utils.get_sbd()

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'img', data_id + '.jpg')
        return read_image(img_file, color=True)

    def _get_annotations(self, i):
        data_id = self.ids[i]
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)

        bbox = []
        for m in mask:
            where = np.where(m)
            y_min = np.min(where[0])
            x_min = np.min(where[1])
            y_max = np.max(where[0])
            x_max = np.max(where[1])
            bbox.append((y_min, x_min, y_max, x_max))
        if len(bbox) > 0:
            bbox = np.array(bbox, dtype=np.float32)
        else:
            bbox = np.zeros((0, 4), dtype=np.float32)
        return bbox, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'cls', data_id + '.mat')
        inst_file = os.path.join(
            self.data_dir, 'inst', data_id + '.mat')
        label_anno = scipy.io.loadmat(label_file)
        label_img = label_anno['GTcls']['Segmentation'][0][0].astype(np.int32)
        inst_anno = scipy.io.loadmat(inst_file)
        inst_img = inst_anno['GTinst']['Segmentation'][0][0].astype(np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img
