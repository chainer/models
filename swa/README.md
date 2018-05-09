# Stochastic Weight Averaging (SWA)
Stochastic Weight Averaging (SWA) implementation in Chainer.

Original paper: [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)

Author's implementation: [https://github.com/timgaripov/swa](https://github.com/timgaripov/swa)

# Introduction

SWA is a training method which can be used for conventional SGD. SWA improve generalization, so it enhances the accuracy of test data. And importantly SWA has almost no overhead compared to SGD. According to the paper, SWA converges more quickly than SGD, and to wider optima.

The fundamental idea of SWA is to average multiple samples produced by SGD with a modified learning rate schedule. For example, you train the model for 100 epochs using SGD. After that, you continue to train the model in the same way but gather the weights every epoch for averaging (actually you compute the running average). It's very simple! Of course, it's straightforward to implement for Chainer.

# Requirement

- Python 3.5.1+
- [Chainer 4.0.0+](https://github.com/pfnet/chainer)
- [CuPy 4.0.0+](https://cupy.chainer.org/)

# Usage
The code in this repository implements both SWA and conventional SGD training.

To run SWA use the following command:

```bash
python3 train.py --dataset=<DATASET> \
                 --model=<MODEL> \
                 --epoch=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --swa \
                 --swa_start=<SWA_START> \
                 --swa_lr=<SWA_LR>
```


## Parameters:

* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR100)
* ```MODEL``` &mdash; DNN model name:
    - VGG16
    - PreResNet110
    - WideResNet28x10
* ```EPOCH``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash; SWA learning rate (default: 0.05)


## Examples

```bash
#VGG16
python train.py --dataset=CIFAR100 --model=VGG16 --epoch=200 --lr_init=0.05 --wd=5e-4 # SGD
python train.py --dataset=CIFAR100 --model=VGG16 --epoch=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets
 
#PreResNet110
python train.py --dataset=CIFAR100 --model=PreResNet110 --epoch=150 --lr_init=0.1 --wd=3e-4 # SGD 
python train.py --dataset=CIFAR100 --model=PreResNet110 --epoch=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05 # SWA 1.5 Budgets

#WideResNet28x10 
python train.py --dataset=CIFAR100 --model=WideResNet28x10 --epoch=200 --lr_init=0.1 --wd=5e-4 # SGD 
python train.py --dataset=CIFAR100 --model=WideResNet28x10 --epoch=300 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 # SWA 1.5 Budgets
```

# Results

Test accuracy (%) of SGD and SWA on CIFAR-100 for different training budgets. 
For each model the _Budget_ is defined as the number of epochs required to train the model with the conventional SGD procedure.

| DNN (Budget)              |  SGD         | SWA 1 Budget | SWA 1.25 Budgets | SWA 1.5 Budgets |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (200)               | 72.22 | 73.65 | 73.88     | 74.09    |
| PreResNet110 (150)        | 78.26 | - | -     | -    |
| WideResNet28x10 (200)     | 81.61 | 82.14 |  82.28   | -    |



### Comparizon with the paper results


| DNN (Budget)              |  SGD         | SWA 1 Budget | SWA 1.25 Budgets | SWA 1.5 Budgets |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (200)               | 72.55 ± 0.10 | 73.91 ± 0.12 | 74.17 ± 0.15     | 74.27 ± 0.25    |
| PreResNet110 (150)        | 78.49 ± 0.36 | 79.77 ± 0.17 | 80.18 ± 0.23     | 80.35 ± 0.16    |
| WideResNet28x10 (200)     | 80.82 ± 0.23 | 81.46 ± 0.23 | 81.91 ± 0.27     | 82.15 ± 0.27    |

