import numpy as np


def vis_video(imgs, repeat=False, ax=None, fig=None):
    """Visualize a video

    Returns:
        Animation
        
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

    img_ax = ax.imshow(imgs[0].transpose((1, 2, 0)).astype(np.uint8))

    def init():
        return img_ax,

    def update(img):
        img = img.transpose((1, 2, 0)).astype(np.uint8)
        img_ax.set_data(img)
        return img_ax,

    fps = 60
    ani = FuncAnimation(
        fig, update, frames=imgs[1:],
        init_func=init, blit=True, interval=1000/fps,
        repeat=repeat)

    return ani
