from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

import json
import numpy as np
import os

from chainercv.utils import read_image


def get_axis_aligned_bb(ply):
    cy = np.mean(ply[:, 0])
    cx = np.mean(ply[:, 1])
    y1 = np.min(ply[:, 0])
    x1 = np.min(ply[:, 1])
    y2 = np.max(ply[:, 0])
    x2 = np.max(ply[:, 1])
    A1 = np.linalg.norm(ply[0] - ply[1]) * np.linalg.norm(ply[1] - ply[2])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    y_min = cy - h / 2
    x_min = cx - w / 2
    y_max = y_min + h
    x_max = x_min + w
    return np.array([y_min, x_min, y_max, x_max], dtype=np.float32)

def old_get_axis_aligned_bb(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = len(region)
    if nv == 8:
        region = np.array(region)
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    
    y_min = cy - h / 2
    x_min = cx - w / 2
    y_max = y_min + h
    x_max = x_min + w
    return np.array([y_min, x_min, y_max, x_max], dtype=np.float32)


def get_min_max_bb(region):
    x1, y1, x2, y2, x3, y3, x4, y4 = region
    y_min = np.min((y1, y2, y3, y4))
    x_min = np.min((x1, x2, x3, x4))
    y_max = np.max((y1, y2, y3, y4))
    x_max = np.max((x1, x2, x3, x4))
    return y_min, x_min, y_max, x_max


class VOTTrackingDataset(GetterDataset):

    def __init__(self, data_dir):
        super(VOTTrackingDataset, self).__init__()

        self.data_dir = data_dir

        self.img_dir = os.path.join(self.data_dir, 'VOT2018')

        anno_path = os.path.join(self.data_dir, 'VOT2018.json')
        self.annos = json.load(open(anno_path))

        self.video_names = sorted(list(self.annos.keys()))
        index_to_video_id = []
        index_to_frame_id = []
        for name in self.video_names:
            n_img = len(self.annos[name]['img_names'])
            index_to_video_id += [self.video_names.index(name)] * n_img
            index_to_frame_id += list(range(n_img))
        self.index_to_video_id = np.array(index_to_video_id, dtype=np.int32)
        self.index_to_frame_id = np.array(index_to_frame_id, dtype=np.int32)

        self.add_getter('img', self._get_img)
        self.add_getter(['video_id', 'frame_id'], self._get_ids)
        self.add_getter('bbox', self._get_bbox)
        self.add_getter('poly', self._get_poly)

    def __len__(self):
        return len(self.index_to_video_id)

    def _get_ids(self, i):
        video_id = self.index_to_video_id[i]
        frame_id = self.index_to_frame_id[i]
        return video_id, frame_id

    def _get_img(self, i):
        video_id, frame_id = self._get_ids(i)
        anno = self.annos[self.video_names[video_id]]
        img_path = os.path.join(self.img_dir, anno['img_names'][frame_id])
        return read_image(img_path)

    def _get_bbox(self, i):
        video_id, frame_id = self._get_ids(i)
        anno = self.annos[self.video_names[video_id]]
        bb = get_axis_aligned_bb(self._get_poly(i)[0])
        return bb[None]

    def _get_poly(self, i):
        video_id, frame_id = self._get_ids(i)
        anno = self.annos[self.video_names[video_id]]
        poly = np.array(anno['gt_rect'][frame_id])
        if len(poly) == 8:
            poly = poly.reshape((4, 2))[:, ::-1]  # y,x
        elif len(poly) == 4:
            x = poly[0]
            y = poly[1]
            w = poly[2]
            h = poly[3]
            poly = np.array((
                (y, x),
                (y, x + w),
                (y + h, x + w),
                (y + h, x)), dtype=np.float32)
        return poly[None]


if __name__ == '__main__':
    from chainercv.visualizations import vis_bbox
    from chainercv.visualizations import vis_point
    import matplotlib.pyplot as plt

    dataset = VOTTrackingDataset('../../data')
    print(dataset.video_names[14])
    img, bbox, poly = dataset.slice[:, ['img', 'bbox', 'poly']][4322]
    vis_bbox(img, bbox)
    plt.savefig('b.png')
    vis_point(img, poly)
    plt.savefig('a.png')
