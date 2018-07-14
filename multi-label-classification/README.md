# Multi-label Image Classification

Chainer implementation of multi-label image classification.
Currently, the training and evaluation is conducted with ResNet50 and PASCAL VOC07.

## Dependency

- Chainer==4.2.0
- CuPy==4.2.0
- ChainerCV==0.10.0

## Train
```
$ python train.py [--gpu <gpu>] [--batchsize <bsize>] [--out <out>]
```

Entire training took 66 minutes on single P100.

## Evaluation
```
$ python eval_voc07.py [--gpu <gpu>] [--pretrained-model <path>]
```

## Demo
```
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Performance
The mAP of the network trained with VOC07 train-val and evaluated on VOC07 test.

```
mAP: 0.861481
aeroplane: 0.949971
bicycle: 0.936831
bird: 0.899449
boat: 0.895450
bottle: 0.600697
bus: 0.852649
car: 0.936476
cat: 0.930157
chair: 0.708775
cow: 0.843032
diningtable: 0.792954
dog: 0.909126
horse: 0.944342
motorbike: 0.912632
person: 0.969591
pottedplant: 0.713204
sheep: 0.865586
sofa: 0.749513
train: 0.963491
tvmonitor: 0.855691
mAP: 0.861481
aeroplane: 0.949971
bicycle: 0.936831
bird: 0.899449
boat: 0.895450
bottle: 0.600697
bus: 0.852649
car: 0.936476
cat: 0.930157
chair: 0.708775
cow: 0.843032
diningtable: 0.792954
dog: 0.909126
horse: 0.944342
motorbike: 0.912632
person: 0.969591
pottedplant: 0.713204
sheep: 0.865586
sofa: 0.749513
train: 0.963491
tvmonitor: 0.855691
```


## References
1. Maksim Lapin, Matthias Hein, and Bernt Schiele. "Analysis and Optimization of Loss Functions for Multiclass, Top-k, and Multilabel Classification" In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.
The row "Multi-label" of Table 10 is particularly relevant [link](https://arxiv.org/pdf/1612.03663.pdf).
