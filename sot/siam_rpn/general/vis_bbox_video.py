import numpy as np


def bbox_to_patch(bbox, patch=None):
    import matplotlib.pyplot as plt
    if bbox is None:
        return patch

    out_patch = []
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]

        if patch is None:
            out_patch.append(
                plt.Rectangle(
                xy, width, height, fill=False))
        else:
            patch[i].set_xy(xy)
            patch[i].set_width(width)
            patch[i].set_height(height)

            out_patch.append(patch[i])

    return out_patch


def vis_bbox_video(imgs, bboxes, repeat=False, ax=None, fig=None):
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
    assert len(imgs) == len(bboxes)

    img_ax = ax.imshow(imgs[0].transpose((1, 2, 0)).astype(np.uint8))

    patch = bbox_to_patch(bboxes[0])
    for p in patch:
        ax.add_patch(p)

    def init():
        return img_ax,

    def update(data):
        img, bbox = data

        img = img.transpose((1, 2, 0)).astype(np.uint8)
        img_ax.set_data(img)
        bbox_to_patch(bbox, patch)
        return img_ax,

    fps = 60
    ani = FuncAnimation(
        fig, update, frames=zip(imgs[1:], bboxes[1:]),
        init_func=init, blit=True,
        interval=1000/fps,
        repeat=repeat)

    return ani


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from vot_tracking_dataset import VOTTrackingDataset
    data_dir = '../../data/'

    dataset = VOTTrackingDataset(data_dir)
    imgs = list(dataset.slice[:, 'img'][:100])
    bboxes = list(dataset.slice[:, 'bbox'][:100])
    ani = vis_bbox_video(imgs, bboxes, repeat=True)
    # For interactive visualization
    plt.show()
    ani.save('a.mp4', writer='ffmpeg', fps=30)
