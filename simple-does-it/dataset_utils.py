from __future__ import division

import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import \
    ConcatenatedDataset
from chainercv.transforms import resize

import PIL

from sbd_bbox_dataset import SBDBboxDataset
from voc_semantic_segmentation_with_bbox_dataset import \
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


def resize_contain(img, size, fill=0, interpolation=PIL.Image.BILINEAR,
                   return_param=False):
    """Resize the image to fit in the given area while keeping aspect ratio.

    If both the height and the width in :obj:`size` are larger than
    the height and the width of the :obj:`img`, the :obj:`img` is placed on
    the center with an appropriate padding to match :obj:`size`.

    Otherwise, the input image is scaled to fit in a canvas whose size
    is :obj:`size` while preserving aspect ratio.

    Args:
        img (~numpy.ndarray): An array to be transformed. This is in
            CHW format.
        size (tuple of two ints): A tuple of two elements:
            :obj:`height, width`. The size of the image after resizing.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            If it is :class:`numpy.ndarray`,
            its shape should be :math:`(C, 1, 1)`,
            where :math:`C` is the number of channels of :obj:`img`.
        return_param (bool): Returns information of resizing and offsetting.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of resizing.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_offset** (*int*): The y coodinate of the top left corner of\
            the image after placing on the canvas.
        * **x_offset** (*int*): The x coordinate of the top left corner\
            of the image after placing on the canvas.
        * **scaled_size** (*tuple*): The size to which the image is scaled\
            to before placing it on a canvas. This is a tuple of two elements:\
            :obj:`height, width`.

    """
    C, H, W = img.shape
    out_H, out_W = size
    scale_h = out_H / H
    scale_w = out_W / W
    scale = min(min(scale_h, scale_w), 1)
    scaled_size = (int(H * scale), int(W * scale))
    if scale < 1:
        img = resize(img, scaled_size, interpolation=interpolation)
    y_slice, x_slice = _get_pad_slice(img, size=size)
    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape((-1, 1, 1))
    out_img[:, y_slice, x_slice] = img

    if return_param:
        param = {'y_offset': y_slice.start, 'x_offset': x_slice.start,
                 'scaled_size': scaled_size}
        return out_img, param
    else:
        return out_img


def _get_pad_slice(img, size):
    """Get slices needed for padding.

    Args:
        img (~numpy.ndarray): This image is in format CHW.
        size (tuple of two ints): (max_H, max_W).
    """
    _, H, W = img.shape

    if H < size[0]:
        margin_y = (size[0] - H) // 2
    else:
        margin_y = 0
    y_slice = slice(margin_y, margin_y + H)

    if W < size[1]:
        margin_x = (size[1] - W) // 2
    else:
        margin_x = 0
    x_slice = slice(margin_x, margin_x + W)

    return y_slice, x_slice


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
