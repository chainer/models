# chainer-nncompress: Chainer Implementations of Embedding Quantization 

Chainer port of NeuralCompressor (nncompress)

Original paper: [Compressing Word Embeddings via Deep Compositional Code Learning](https://openreview.net/pdf?id=BJRZzFlRb)

Author's implementation: [https://github.com/zomux/neuralcompressor](https://github.com/zomux/neuralcompressor)

## Requirements

- Python 3.5.1+
- Chainer 4.1.0+
- CuPy 4.1.0+
- logzero

## Usage

1. Prepare data

```
> bash scripts/download_glove_data.sh
```

2. Convert the GloVe embeddings to numpy format

```
> python scripts/convert_glove2numpy.py data/glove.6B.300d.txt
```

3. Train the embedding quantization model

```
> python train.py -b 128 -gpu 0 -O Adam --M 8 --K 8 --tau 1.0 --input-matrix ./data/glove.6B.300d.npy
```

```
...
63          198000      16.7019     17.0396               0.788086    0.788811
63          198000      16.7019     17.0396               0.788086    0.788811
63          199000      17.7445     17.0096               0.777498    0.786644
63          199000      17.7445     17.0096               0.777498    0.786644
64          200000      16.878      17.0317               0.789763    0.787662
64          200000      16.878      17.0317               0.789763    0.787662
[I 180528 09:42:24 train:114] Training complete!!
[I 180528 09:42:24 resource:113] EXIT TIME: 20180528 - 09:42:24
[I 180528 09:42:24 resource:115] Duration: 0:26:48.503415
...
```

4. Export the word codes and the codebook matrix

```
> python decode.py --gpu 0 --model ./result/20180528_091536_model_seed_0_optim_Adam_tau_1.0_batch_128_M_8_K_8/iter_199000.npz --vocab data/glove.6B.300d.word --embed ./data/glove.6B.300d.npy
```

It will generate two files in result dir:
- iter_199000.npz.codebook.npy
    - this contains parameters of the compressed embedding matrix
- iter_199000.npz.codes
    - this contains code representation of each word

6. Check the codes

```
> head -100 iter_199000.npz.codes
```

```
...
human   5 2 7 2 4 7 4 1
india   2 2 7 0 4 1 0 6
defense 2 2 7 0 4 4 0 1
asked   7 2 7 2 4 4 2 6
total   2 0 7 4 4 3 0 5
october 2 0 7 4 4 7 7 7
players 0 0 7 6 4 4 0 6
bill    7 2 7 2 4 7 2 0
...
```

You can see that similar words have similar code representations.
```
dog     2 3 7 6 4 4 2 1
dogs    2 3 7 6 4 4 2 6
cat     4 3 7 1 4 7 2 7
penguin 6 3 7 2 6 7 2 7
```

## Citation

```
@inproceedings{shu2018compressing,
title={Compressing Word Embeddings via Deep Compositional Code Learning},
author={Raphael Shu and Hideki Nakayama},
booktitle={International Conference on Learning Representations (ICLR)},
year={2018},
url={https://openreview.net/forum?id=BJRZzFlRb},
}
```

