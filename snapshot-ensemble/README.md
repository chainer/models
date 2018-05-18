# Snapshot Ensembles
Implementation of Snapshot Ensemble in Chainer.

Original paper: [SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE](https://arxiv.org/abs/1704.00109)

# Introduction

Snapshot Ensemble is a method to train multiple models which can be ensembled. It improves generalization by the ensemble. And importantly it achieves that at no additional training cost. Snapshot Ensemble train a single model, and along its optimization, it saves the model parameters at certain epoch. Therefore the weights being "snapshots" of the model.

The key idea of  Snapshot Ensemble is to find multiple "good" local optima along its optimization. To achieve that, the authors use cosine annealing cycles as the learning rate schedule. They found that there exist multiple local minima when training a model using the cosine annealing cycles. 

It can be  described using the following image from the paper:

# Requirement

- Python 3.5.1+
- [Chainer 4.0.0+](https://github.com/pfnet/chainer)
- [CuPy 4.0.0+](https://cupy.chainer.org/)

# Usage
The code in this repository implements both Snapshot Ensemble and conventional SGD training.

To train bySnapshot Ensemble use the following command:

```bash
python3 train.py --dataset=<DATASET> \
                 --model=<MODEL> \
                 --epoch=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --se \
                 --se_cycle=<SE_CYCLE> \
```


Parameters:

* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR100)
* ```MODEL``` &mdash; DNN model name:
    - VGG16
    - PreResNet110
    - WideResNet28x10
* ```EPOCH``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; split the training process into N cycles,each of which starts with a large learning rate (default: 5)


To evaluate models trained by Snapshot Ensemble, use the following command:

```bash
python3 eval.py --dataset=<DATASET> \
                 --model=<MODEL> \
                 --out=<OUT>
```

Parameters:

* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR100)
* ```MODEL``` &mdash; DNN model name:
* ```OUT``` &mdash; the directory from which read the snapshot files.

## Examples

```bash

#train
python train.py --dataset=CIFAR100 --model=PreResNet110 --lr_init=0.1 # Baseline (Single)
python train.py --dataset=CIFAR100 --model=PreResNet110 --lr_init=0.1 --out=result_resnet --se # snapshot ensemble

#eval
python eval.py --dataset=CIFAR100 --model=PreResNet110 --out=result_resnet
```

# Results

Test accuracy (%) of CIFAR-100 for different training methods by our codes. 

| Model              |  Single         | NoCycle Snapshot Ensemble | Snapshot Ensemble |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| PreResNet110        | 67.91 | 69.70 | 70.68  |


- `Single` is the performance of the single model.
- `Snapshot Ensemble` is the performance of Snapshot Ensemble. 
- `NoCycle Snapshot Ensemble` is the same as the `Snapshot Ensemble`, but uses the same learning rate schedule as the `Single` model.