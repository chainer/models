import numpy as np
import os
import xml.etree.ElementTree as ET

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCSemanticSegmentationWithBboxDataset(GetterDataset):

    """Semantic segmentation dataset for PASCAL `VOC2012`_.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='auto', split='aug'):
        super(VOCSemanticSegmentationWithBboxDataset, self).__init__()

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2012', split)

        if split == 'aug':
            file_dir = os.path.dirname(os.path.realpath(__file__))
            f = open(os.path.join(file_dir, 'splits/voc_trainaug.txt'))
            self.ids = []
            for l in f.readlines():
                path = l.split()[0]
                self.ids.append(os.path.split(os.path.splitext(path)[0])[1])
        elif split == 'val':
            id_list_file = os.path.join(
                data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
            self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter('label_map', self._get_label)
        self.add_getter(('bbox', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, 'SegmentationClass', self.ids[i] + '.png')
        label = read_image(label_path, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]

    def _get_annotations(self, i):
        use_difficult = False
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.zeros((0, 4), dtype=np.float32)
            label = np.zeros((0,), dtype=np.int32)
        return bbox, label
