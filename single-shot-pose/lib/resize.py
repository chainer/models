import numpy as np
import PIL


def resize(img, size):
    # Same resize method as torchvision
    H, W = size
    img = img.transpose(1, 2, 0).astype(np.uint8)
    p = PIL.Image.fromarray(img, mode='RGB')
    p = np.array(p.resize((W, H)))
    return p.transpose(2, 0, 1).astype(np.float32)


if __name__ == '__main__':
    from linemod_dataset import LinemodDataset
    from chainercv.visualizations import vis_image
    import matplotlib.pyplot as plt

    dataset = LinemodDataset('..')
    img, _, _ = dataset[0]
    img = resize(img, (543, 543))
    vis_image(img)
    plt.show()
