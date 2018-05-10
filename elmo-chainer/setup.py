#!/usr/bin/python
import setuptools

setuptools.setup(
    name='bilm',
    version='0.1',
    description='ELMo by Chainer',
    author='Sosuke Kobayashi',
    author_email='sosk@preferred.jp',
    url='http://github.com/chainer/models/elmo-chainer',
    packages=[
        'bilm'
    ],
    tests_require=[],
    zip_safe=False,
    entry_points='',
    install_requires=[
        'chainer>=2.0',
        'numpy',
        'tqdm',
        'h5py'
    ])
