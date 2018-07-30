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

## Dependency

- Chainer==4.2.0
- CuPy==4.2.0
- ChainerCV==0.10.0

## Usage

##### Data preparation

Download preprocessed LINEMOD data.

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

##### Evaluation

```bash
$ python eval_linemod.py [--gpu <gpu>] [--pretrained-model <model_path>]

e.g.,
$ python eval_linemod.py --gpu 0 --pretrained-model ssp_yolo_v2.npz
```

Here are the scores. The scores in each block corresponds to `PyTorch score[1] and Chainer converted weight`.

| Object | 2D projection | 3D transform | Mean pixel error |
|:-:|:-:|:-:|:-:|
| Ape | 94.48 / 94.48 | 28.00 / 28.00 | 2.8339 / 2.8339  |


## References
1. https://github.com/Microsoft/singleshotpose (the original author's implementation in PyTorch)
