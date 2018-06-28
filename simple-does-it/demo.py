import argparse

import matplotlib.pyplot as plt

import chainer
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation

from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors

from wss.datasets.dataset_utils import get_sbd_augmented_voc
from wss.model import get_pspnet_resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('--input-size', type=int, default=448)
    args = parser.parse_args()

    label_names = voc_semantic_segmentation_label_names
    colors = voc_semantic_segmentation_label_colors
    n_class = len(label_names)

    input_size = (args.input_size, args.input_size)
    model = get_pspnet_resnet50(n_class)
    chainer.serializers.load_npz(args.pretrained_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    dataset = get_sbd_augmented_voc()
    for i in range(1, 100):
        img = dataset[i][0]

        # img = read_image(args.image)
        labels = model.predict([img])
        label = labels[0]

        from chainercv.utils import write_image
        write_image(
            label[None], '{}.png'.format(i))

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        vis_image(img, ax=ax1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2, legend_handles = vis_semantic_segmentation(
            img, label, label_names, colors, ax=ax2)
        ax2.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)

        plt.show()


if __name__ == '__main__':
    main()
