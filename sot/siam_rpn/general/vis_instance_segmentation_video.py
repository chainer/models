import numpy as np

from chainercv.visualizations.colormap import voc_colormap
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox


def _get_overlayed_image(img, mask, instance_colors):
    alpha = 0.5
    bbox = mask_to_bbox(mask)

    _, H, W = mask.shape
    canvas_img = np.zeros((H, W, 3), dtype=np.uint8)
    canvas_img[:] = img.transpose((1, 2, 0)).astype(np.uint8)
    for i, (bb, msk) in enumerate(zip(bbox, mask)):
        # The length of `colors` can be smaller than the number of instances
        # if a non-default `colors` is used.
        color = instance_colors[i % len(instance_colors)]
        # rgba = np.append(color, alpha * 255)
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max > y_min and x_max > x_min:
            canvas_img[msk] = color * alpha

    return canvas_img


def vis_instance_segmentation_video(imgs, masks, repeat=False, ax=None, fig=None):
    """Vis video with bounding boxes


    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if fig is None:
        fig = plt.figure()
        if ax is not None:
            raise ValueError('when ax is not None, fig should be not None')

    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    assert len(imgs) > 0
    assert len(imgs) == len(masks)

    n_inst = 10
    instance_colors = voc_colormap(list(range(1, n_inst + 1)))
    instance_colors = np.array(instance_colors)

    overlayed_img = _get_overlayed_image(imgs[0], masks[0], instance_colors)
    img_ax = ax.imshow(overlayed_img)

    def init():
        return img_ax,

    def update(data):
        img, mask = data

        overlayed_img = _get_overlayed_image(img, mask, instance_colors)
        img_ax.set_data(overlayed_img)
        return img_ax,

    fps = 60
    ani = FuncAnimation(
        fig, update, frames=zip(imgs[1:], masks[1:]),
        init_func=init, blit=True,
        interval=1000/fps,
        repeat=repeat)

    return ani
