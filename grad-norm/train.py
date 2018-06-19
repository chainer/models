import argparse
import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F

from dataset import RegressionDataset

from model import RegressionChain
from model import RegressionTrainChain


def main():
    parser = argparse.ArgumentParser(description='GradNorm')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--n-iter', '-it', type=int, default=5000)
    parser.add_argument('--mode', '-m', choices=('grad_norm', 'equal_weight'),
                        default='grad_norm')
    args = parser.parse_args()

    np.random.seed(123)
    sigmas = [1, 10]
    n_task = len(sigmas)
    epsilons = np.random.normal(
        scale=3.5, size=(n_task, 100, 250)).astype(np.float32)
    dataset = RegressionDataset(sigmas, epsilons)

    model = RegressionTrainChain(RegressionChain(n_task))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=1e-2)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(dataset, 200)

    xp = model.xp
    weights = []
    task_losses = []
    loss_ratios = []
    final_layer_names = ['task_{}'.format(i) for i in range(n_task)]
    for t in range(args.n_iter):
        batch = train_iter.next()
        x, ts = chainer.dataset.convert.concat_examples(batch, device=args.gpu)

        task_loss = model(x, ts)
        weighted_task_loss = model.weight * task_loss
        if t == 0:
            initial_task_loss = task_loss.data
        loss = F.mean(weighted_task_loss)
        model.cleargrads()
        loss.backward()
        # Ignore a gradient to the coefficient vector, which
        # is computed from the standard loss.
        model.weight.cleargrad()
        if args.mode == 'grad_norm':
            # Use |\nabla_W w_i * L_i | = w_i |\nabla_W L_i|
            gygw_norms = []
            for i, layer_name in enumerate(final_layer_names):
                l = getattr(model.model, layer_name)
                gygw = chainer.grad([task_loss[i]], [l.W])[0].data
                gygw_norms.append(xp.linalg.norm(gygw))
            gygw_norms = xp.stack(gygw_norms)
            norms = model.weight * gygw_norms

            alpha = 0.16
            mean_norm = xp.mean(norms.data)
            loss_ratio = task_loss.data / initial_task_loss
            inverse_train_rate = loss_ratio / xp.mean(loss_ratio)

            diff = norms - (inverse_train_rate ** alpha) * mean_norm
            grad_norm_loss = F.mean(F.absolute(diff))
            grad_norm_loss.backward()

            # For debugging purpose only
            # from chainer import computational_graph
            # import os
            # cg = computational_graph.build_computational_graph(
            #     [grad_norm_loss]).dump()
            # with open('grad_weight_loss_cg', 'w') as f:
            #     f.write(cg)

        optimizer.update()

        # Renormalize
        normalize_coeff = n_task / xp.sum(model.weight.data)
        model.weight.data[:] = model.weight.data * normalize_coeff

        # Record
        task_losses.append(chainer.backends.cuda.to_cpu(task_loss.data))
        loss_ratios.append(np.mean(task_losses[-1] / task_losses[0]))
        weights.append(chainer.backends.cuda.to_cpu(model.weight.data))

        if t % 100 == 0:
            print('{}/{}:  loss_ratio={}, weights={} task_loss={}'.format(
                t, args.n_iter, loss_ratios[-1], model.weight.data, task_loss.data))
    task_losses = np.array(task_losses)
    weights = np.array(weights)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title('loss (task 0)')
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title('loss (task 1)')
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_title('sum of normalized losses')
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title('change of weights over time')
    ax1.plot(task_losses[:, 0])
    ax2.plot(task_losses[:, 1])
    ax3.plot(loss_ratios)
    ax4.plot(weights[:, 0])
    ax4.plot(weights[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
