import numpy as np

from chainercv import transforms


def bbox_to_multi_label(in_data):
    img, bbox, label = in_data
    return img, np.unique(label)


class BatchTransform(object):

    def __init__(self, mean):
        self._mean = mean
        self._min_size = 300
        self._max_size = 500
        self._stride = 224

    def __call__(self, imgs):
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self._mean
            resized_imgs.append(img)

        size = np.array([img.shape[1:] for img in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        return x
