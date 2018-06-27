import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import \
    ConcatenatedDataset

import PIL

from wss.datasets.resize_contain import resize_contain
from wss.datasets.sbd_bbox_dataset import SBDBboxDataset
from wss.datasets.voc_semantic_segmentation_with_bbox_dataset import \
    VOCSemanticSegmentationWithBboxDataset


def resize_contain_img_and_label_map(img, label_map):
    img = resize_contain(img, (448, 448))
    label_map, param = resize_contain(
        label_map[None], (448, 448), interpolation=PIL.Image.NEAREST,
        return_param=True)
    label_map = label_map[0]
    resized_label_map = label_map
    label_map = np.ones_like(label_map) * -1
    y_slice = slice(param['y_offset'],
                    param['y_offset'] + param['scaled_size'][0])
    x_slice = slice(param['x_offset'],
                    param['x_offset'] + param['scaled_size'][1])

    label_map[y_slice, x_slice] = resized_label_map[y_slice, x_slice]
    return img, label_map


class SimpleDoesItTransform(object):

    def __init__(self, mean=None):
        self.mean = mean

    def __call__(self, in_data):
        if len(in_data) == 3:
            img, bbox, instance_label = in_data
            pred_label_map = None
        else:
            img, bbox, instance_label, pred_label_map = in_data

        # Adjust to semantic segmentation label names
        instance_label = instance_label + 1

        if self.mean is not None:
            img = img - self.mean
        label_map = np.zeros(img.shape[1:], dtype=np.int32)
        bbox = bbox.astype(np.int32)
        # Paste smaller bounding boxes on top of larger ones
        area = [(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in bbox]
        for index in np.argsort(area)[::-1]:
            bb = bbox[index]
            lb = instance_label[index]
            if pred_label_map is None:
                label_map[bb[0]:bb[2], bb[1]:bb[3]] = lb
            else:
                cropped_mask = pred_label_map[bb[0]:bb[2], bb[1]:bb[3]] == lb

                # take into the fact that bbox includes mask
                iou = cropped_mask.sum() / area[index]
                if iou < 0.3:
                    # reset to initial label
                    label_map[bb[0]:bb[2], bb[1]:bb[3]] = lb
                else:
                    label_map[bb[0]:bb[2], bb[1]:bb[3]] = cropped_mask * lb

        img, label_map = resize_contain_img_and_label_map(img, label_map)
        return img, label_map


def get_sbd_augmented_voc():
    voc = VOCSemanticSegmentationWithBboxDataset(
        split='aug').slice[:, ['img', 'bbox', 'label']]
    sbd = SBDBboxDataset()
    dataset = ConcatenatedDataset(voc, sbd)
    return dataset


if __name__ == '__main__':
    from chainercv.visualizations import vis_bbox
    from chainercv.visualizations import vis_semantic_segmentation
    from chainercv.chainer_experimental.datasets.sliceable import TransformDataset 
    import matplotlib.pyplot as plt
    import numpy as np

    voc_segm = VOCSemanticSegmentationWithBboxDataset(
        split='aug')
    dataset = VOCSemanticSegmentationWithBboxDataset(
        split='aug').slice[:, ['img', 'bbox', 'label']]
    # transformed = TransformDataset(
    #     dataset, ('img', 'label_map'), grabcut_transform)
    transformed = TransformDataset(
        dataset, ('img', 'label_map'), SimpleDoesItTransform())

    indices = np.random.choice(np.arange(len(voc_segm)), size=(10,))
    for index in indices:
        img, label_map, bbox, label = voc_segm[index]

        vis_bbox(img, bbox, label)
        plt.show()
        # see doc for better visualization
        vis_semantic_segmentation(img, label_map)
        plt.show()

        img, label_map = transformed[index]
        vis_semantic_segmentation(img, label_map, alpha=0.6, ignore_label_color=(255, 0, 0))
        plt.show()
