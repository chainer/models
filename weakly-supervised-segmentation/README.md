# Weakly Supervised Segmentation

Chainer implementation of the baseline method of [Simple Does It](https://arxiv.org/abs/1603.07485) (`Box` of Table 1 described in Section 3.1).

Citation:

```
@inproceedings{Chen:2018,
  author={Anna Khoreva, Rodrigo Benenson, Jan Hosang, Matthias Hein, Bernt Schiele},
  title={Simple Does It: Weakly Supervised Instance and Semantic Segmentation},
  booktitle={CVPR},
  year={2017}
}
```

## Usage

```bash
$ python train.py [--gpu <gpu>]
```

## Relation to the implementation of the paper
These are known difference from the original paper:
1. PSPNet with ResNet50 backbone is used instead of DeepLab with VGG16.
2. Batchsize and image size are different.
3. We do not use DenseCRF to improve masks.
4. No data augmentation is used by us.

