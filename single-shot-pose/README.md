# Real-Time Seamless Single Shot 6D Object Pose Prediction

Chainer implementation of [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/abs/1711.08848) by Yusuke Niitani.

Citation:
```
@inproceedings{Chen:2018,
  author={Bugra Tekin, Sudipta N. Sinha, Pascal Fua},
  title={Real-Time Seamless Single Shot 6D Object Pose Prediction},
  booktitle={CVPR},
  year={2018}
```

## Tested environment

- Chainer==4.2.0
- CuPy==4.2.0
- ChainerCV==0.10.0

## Usage

##### Data preparation

Download preprocessed LINEMOD data, which is necessary for training and evaluation.

```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
```

The Chainer weights converted from the weight of the original author is found [here](https://drive.google.com/open?id=134AGqFgQHwwVwO3t1sKRGY02pFV_0usd).
This model predicts the pose of `ape` from the LINEMOD dataset.
Also, you can convert PyTorch weights to Chainer weights by yourself by following the instruction in `conversion`.

##### Demo

```bash
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] <image>
```


##### Weight conversion

See [here](https://github.com/chainer/models/tree/master/single-shot-pose/conversion)

##### Train

```bash
$ python train.py [--gpu <gpu>] [--pretrained-model <model_path>] [--out <out>] [--batchsize <bsize>] [--lr <lr>] <object_name>

e.g.,
$ python train.py --pretrained-model conversion/ssp_yolo_v2_linemod_ape_init_converted.npz --gpu 0 --out result/ape ape

```

###### Difference with the original implementation
1. batchsize is changed from 32 to 24
2. Maximum size of image during training is smaller
3. No color related data augmentation
4. Image scale is `[0, 255]`. This changes amount of weight decay for the first layer.

##### Evaluation

```bash
$ python eval_linemod.py [--gpu <gpu>] [--pretrained-model <model_path>] <object_name>

e.g.,
$ python eval_linemod.py --gpu 0 --pretrained-model conversion/ssp_yolo_v2_linemod_ape_converted.npz
```

Here are the scores. There are four scores in each block.
From left to right, these four scores are from
1. PyTorch code [1]
2. Chainer converted weight
3. (two phase training) Chainer trained weights starting from the weights pretrained with `conf_loss_scale=0`
4. (one phase training) Chainer trained weights starting from the ImageNet pretrained weights

| Object | 2D projection acc (%) | 3D transform acc (%) | Mean pixel error (px) |
|:-:|:-:|:-:|:-:|
| ape | 94.48 / 94.48 / **97.14** / 96.76  | **28.00** / **28.00** / 27.81 / 27.05  | 2.8339 / 2.8339 / **1.9337** / 1.9860   |
| benchvise | 94.86 / 94.86 / **96.51** / 83.14 | 78.68 / 78.68 / 78.68 / 54.75 | 2.5054 / 2.5054 / **2.5031**/ 3.7296  |

The two phase training is more stable for some objects.


## References
1. https://github.com/Microsoft/singleshotpose (the original author's implementation in PyTorch)
