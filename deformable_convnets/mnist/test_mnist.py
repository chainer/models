import argparse
import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.dataset.convert import concat_examples

from chainercv.datasets import TransformDataset
from chainercv import transforms

from models import Convnet
from models import DeformableConvnet
from scale_transform import transform


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--resume', '-r', default='result/model_iter_9',
                        help='Resume the training from snapshot')
    parser.add_argument('--deformable', '-d', type=int, default=1,
                        help='use deformable convolutions')
    args = parser.parse_args()

    if args.deformable == 1:
        model = DeformableConvnet(10)
    else:
        model = Convnet(10)
    chainer.serializers.load_npz(args.resume, model)

    train, test = chainer.datasets.get_mnist(ndim=3)
    test = TransformDataset(test, transform)

    test_iter = chainer.iterators.SerialIterator(test, batch_size=1,
                                                 repeat=False, shuffle=False)

    threshold = 1
    for i in range(1):
        batch = test_iter.next()
        in_arrays = concat_examples(batch, device=None)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)
        img, label = in_vars
        model(img)
        feat = model.feat
        H, W = feat.shape[2:]
        center = F.sum(feat[:, :, H / 2, W / 2])
        center.grad= np.ones_like(center.data)
        model.zerograds()
        img.zerograd()
        center.backward(retain_grad=True)

        img_grad = img.grad[0]  # (1, 28, 28)

        img_grad_abs = (np.abs(img_grad) / np.max(np.abs(img_grad)) * 255)[0]  # 28, 28
        img_grad_abs[np.isnan(img_grad_abs)] = 0
        y_indices, x_indices = np.where(img_grad_abs > threshold)
        plt.scatter(x_indices, y_indices, c='red')
        vis_img = transforms.chw_to_pil_image(255 * img.data[0])[:, :, 0]
        plt.imshow(vis_img, interpolation='nearest', cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
