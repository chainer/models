#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from chainer import optimizers
import matplotlib.pyplot as plt
import numpy as np

import mdn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--input-dim', '-d', type=int, default=1)
    parser.add_argument('--n-samples', '-n', type=int, default=1000)
    parser.add_argument('--hidden-units', '-u', type=int, default=24)
    parser.add_argument('--gaussian-mixtures', '-m', type=int, default=24)
    parser.add_argument('--epoch', '-e', type=int, default=10000)
    args = parser.parse_args()

    y_data = np.float32(np.random.uniform(-10.5, 10.5, (args.n_samples, args.input_dim)))
    noise = np.random.normal(size=(args.n_samples, args.input_dim))
    x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + noise * 1.0)

    # Plot the target data
    plt.scatter(x_data, y_data, c='r', alpha=0.3)
    plt.savefig('images/target.png')

    # Instantiate a model
    model = mdn.MDN(args.input_dim, args.hidden_units, args.gaussian_mixtures)

    # Prepare an optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Train
    loss_history = []
    iteration = 0
    for epoch in range(args.epoch):
        loss = model.negative_log_likelihood(x_data, y_data)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        iteration += 1
        loss_history.append(float(loss.array))
        if epoch % 100 == 0:
            print('epoch:', epoch, 'iteration:', iteration, 'loss:', float(loss.array))

    # Plot results
    plt.clf()
    plt.plot(loss_history)
    plt.ylabel('negative log likelihood')
    plt.xlabel('iteration')
    plt.savefig('images/loss.png')

    pred_x_data = np.random.uniform(-15, 15, size=(3000, 1)).astype(np.float32)
    pred_y_data = model.sample(pred_x_data).array

    plt.clf()
    plt.scatter(pred_x_data, pred_y_data, alpha=0.3)
    plt.savefig('images/generated.png')

    plt.clf()
    plt.scatter(x_data, y_data, c='r', alpha=0.3)
    plt.scatter(pred_x_data, pred_y_data, alpha=0.3)
    plt.savefig('images/overlap.png')
