# ELMo-chainer

Chainer implementation of the pretrained biLM used to compute ELMo representations from ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).
The paper reported that the contextual representations provide large improvements for NLP tasks.

This repository is unofficially ported from [tensorflow bilm-tf](https://github.com/allenai/bilm-tf) and [pytorch in AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md). Thank you, [@allenai](https://github.com/allenai) and [@matt-peters](https://github.com/matt-peters).
This Chainer implementation is primarily compatible with them and able to use model files as well as they use.

Citation:

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```

## Install

```
pip install chainer h5py tqdm
python setup.py install
sh download.sh
```


## Usage

You have some choises using ELMo due to caching for acceleration. These usages correspond to the [bilm-tf's description](https://github.com/allenai/bilm-tf#using-pre-trained-models). [pytorch's](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) readme will be also helpful.
You can use either gpu and cpu.

### 1. Compute representations on the fly from raw text using character input

- When you evaluate a model at test time on unseen data (e.g. public SQuAD leaderboard)
- You don't have to know and fix any token vocabulary

c.f. `usage_character.py` or `usage_character_simple.py`

### 2. Precompute and cache the token representations, then compute contextualized representations with the biLSTMs for input data

- When you train or evaluate a model on dynamically changed dataset (e.g. sentences generated by GAN)
- When you train or evaluate a model on too large dataset to cache all contextualized representations
- You have to know the token vocabulary

c.f. `usage_token.py`

### 3. Precompute contextualized representations for an entire dataset and save to a file

- When you train or evaluate a model on a fixed dataset
- You have to know the token vocabulary

c.f. `usage_cache.py` or `bilm_encode_sentences.py`

You can save the biLM representations of dataset (sentence-per-line) as follows:

```
python bilm_encode_sentenses.py -i _sample_dataset_file.txt -o elmo_embeddings.hdf5
```

You can use gpu (id=0) if adding `-g 0`.


You can load them as numpy.ndarray in python

```
embedding_file = 'elmo_embeddings.hdf5'
with h5py.File(embedding_file, 'r') as fin:
    sent_idx = '0'
    sentence_embeddings = fin[sent_idx][...]
    print(sentence_embeddings.shape)
    # shape = (n_lstm_layers=3, sequence_length, embedding_dim)
```


## Note

- Can load model files with the orignal formats.
- Training of weight-sum layers is supported.
- Pretraining of biLM is NOT supported.
- Finetuning of biLM in downstream tasks is possible but disabled by default.
- Excuse: This is modified to work on both py3 and py2 unlike original implementations, but py3 is recommended.
- Docstrings in code are still old, i.e. written for pytorch, but you can almost read them due to similarity with chainer.


### If Using GPU

If you know your cuda version as CUDA 8.0, 9.0 or 9.1, please perform the corresponding installation.

```
pip install cupy-cuda80
pip install cupy-cuda90
pip install cupy-cuda91
```

Otherwise, please

```
pip install cupy
```

### Appendix

Elmo takes as input tokenized sentences.
It does not require lowercasing but requires separating punctuations without words.
A lazy user can perform the punctuation-separation with a simple script as follows:

```
cat _sample_dataset_file_with_punc.txt | python split_sentence_with_punctuations.py
```