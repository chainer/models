import argparse
import copy
import numpy as np
import os
import time

import chainer
from chainer.dataset import convert
from chainer import functions as F
from chainer import serializers
import cupy

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

from logging import getLogger, DEBUG, Formatter, StreamHandler
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
# handler.setFormatter(Formatter('%(asctime)s %(message)s'))
logger.addHandler(handler)

import nets


def visualize(result, name):
    # (N_samples=5, timesteps=200, types)
    result = np.array(result)
    assert result.shape[2] == 2
    # ax = sns.tsplot(data=result, condition=['OptNet', 'Adam'], linestyle='--')

    def trans(series):
        # (N_samples, timesteps)
        x = np.tile(np.arange(series.shape[1]) + 1,
                    (series.shape[0], 1)).flatten()
        y = series.flatten()
        return {'x': x, 'y': y}
    ax = sns.lineplot(label='OptNet', **trans(result[:, :, 0]))
    ax = sns.lineplot(label='Adam', ax=ax, **trans(result[:, :, 1]))
    ax.lines[-1].set_linestyle('-')
    ax.legend()
    plt.yscale('log'), plt.xlabel('steps')
    plt.ylabel('loss'), plt.title('MNIST')
    plt.ylim(0.09, 3.0)
    plt.xlim(1, result.shape[1])
    plt.grid(which='both', alpha=0.6, color='black', linewidth=0.1,
             linestyle='-')
    ax.tick_params(which='both', direction='in')
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    plt.show()
    plt.savefig(name)
    plt.close()


def evaluate_optimizer(args, train, optimizer):
    if isinstance(optimizer, nets.optnets.LSTMOptNet):
        optimizer.release_all()
    device = chainer.get_device(args.gpu)
    device.use()
    n_evaluation_runs = args.evaluation_runs  # 5?
    max_iter_of_meta = args.iter_meta  # 100

    all_losses = []
    for _ in range(n_evaluation_runs):
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        losses = []
        model = nets.images.MLPforMNIST()
        model.to_device(device)
        optimizer.setup(model)

        iteration = 0
        while iteration < max_iter_of_meta:
            # routine
            iteration += 1
            batch = train_iter.next()
            batch = convert.concat_examples(batch, device=device)
            x, t = batch
            with chainer.using_config('train', True):
                loss, acc = model(x, t, get_accuracy=True)
            model.cleargrads()
            loss.backward(retain_grad=False)  # False
            optimizer.update(train_optnet=False)
            losses.append(loss.item())  # log
        all_losses.append(losses)
    # TODO: use losses in only last half iterations?
    last10_mean = np.mean([losses[-10:] for losses in all_losses])
    return last10_mean, all_losses


def pretraining(optimizer):
    logger.info('pretraining')
    copy_grand_opt = copy.deepcopy(optimizer.grand_optimizer)
    losses = []
    for _ in range(10):
        x = optimizer.optnet.xp.random.normal(
            scale=10., size=(10000, 1)).astype('f')
        g = optimizer.optnet.step(x)
        # loss forcing g's sign to be the flip of input's sign
        # theta = theta - c*gradient
        # theta = theta + g
        loss = F.mean(F.clip(g, 0, 100) * (x > 0)
                      + F.clip(-g, 0, 100) * (x < 0))
        optimizer.optnet.cleargrads()
        loss.backward()
        optimizer.meta_update()
        optimizer.optnet.reset_state()
        losses.append(loss.item())
    logger.info('finished pretraining. losses {}'.format(losses))
    optimizer.release_all()
    # reset adam state
    optimizer = nets.optnets.OptimizerByNet(optimizer.optnet, copy_grand_opt)
    return optimizer, copy_grand_opt


def set_seed(seed):
    np.random.seed(seed)
    cupy.random.seed(seed)
    os.environ['CHAINER_SEED'] = str(seed)
    logger.info('set seed {}'.format(seed))


def get_adam_result(args, test_data):
    optimizer = chainer.optimizers.Adam(alpha=args.adam_alpha)
    test_loss, test_all_losses = evaluate_optimizer(args, test_data, optimizer)
    return test_loss, test_all_losses


def train_optimizer(args):
    device = chainer.get_device(args.gpu)
    device.use()

    # prepare test (meta-inference) set from training data
    train, _ = chainer.datasets.get_mnist(ndim=1)
    train, test = chainer.datasets.split_dataset_random(
        train, int(len(train) * 0.5), seed=2400)

    # get results using adam
    set_seed(args.seed)
    adam_test_loss, adam_test_all_losses = get_adam_result(args, test)
    if args.use_adam:
        logger.info('use Adam')
        logger.info('\t'*2 + 'TEST last10mean loss {:.5f}'.format(test_loss))
        return None  # finish

    # make learnable optimizer and its optmizer
    set_seed(args.seed)
    logger.info('use MetaOpt')
    optnet = nets.optnets.LSTMOptNet(out_scale=args.out_scale,
                                     do_preprocess=True)
    optnet.to_device(device)
    grand_opt = chainer.optimizers.Adam(alpha=args.adam_alpha)
    optimizer = nets.optnets.OptimizerByNet(optnet, grand_opt)

    # hyperparams of training
    max_cycle_of_meta = args.cycle_meta  # many
    max_iter_of_meta = args.iter_meta  # 100
    unroll_iters = args.unroll_iters  # 20
    optimizer_path = args.optimizer_path
    best_test_loss = 100000000000  # inf

    # original pretraining:
    # fix bad initialization with anti-update
    # where a model is updated to the direction of gradient.
    # (typically, like SGD, it should be the opposite of gradient.)
    if args.do_pretraining:
        set_seed(args.seed)
        optimizer, grand_opt = pretraining(optimizer)

    # train
    set_seed(args.seed)
    logger.info('training start')
    for i_cycle in range(max_cycle_of_meta):
        # in each cycle,
        # meta-train optimizer through training newly initialized model
        time_cycle_start = time.time()
        meta_losses, loss_logs, acc_logs = [], [], []

        # init model
        model = nets.images.MLPforMNIST()
        model.to_device(device)
        optimizer.setup(model)

        # reset data iterator
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        iteration = 0
        while iteration < max_iter_of_meta:
            iteration += 1
            batch = train_iter.next()
            batch = convert.concat_examples(batch, device=device)
            x, t = batch
            with chainer.using_config('train', True):
                loss, acc = model(x, t, get_accuracy=True)
            model.cleargrads()
            loss.backward(retain_grad=True)
            optimizer.update()

            meta_losses.append(loss)  # stored for meta update
            loss_logs.append(loss.item())  # log
            acc_logs.append(acc)  # log

            # meta update
            if len(meta_losses) >= unroll_iters:
                optimizer.optnet.cleargrads()
                sum(meta_losses).backward(retain_grad=True)
                model.cleargrads()
                optimizer.meta_update()
                meta_losses = []

        iter_per_sec = iteration / (time.time() - time_cycle_start)
        logger.info('  cycle {}\tlast10mean loss {:.5f}\tacc {:.5f}\t({:.2f} i/s)'
                    .format(i_cycle, np.mean(loss_logs[-10:]), np.mean(acc_logs[-10:]), iter_per_sec))

        test_loss, test_all_losses = evaluate_optimizer(
            args, test, copy.deepcopy(optimizer))
        logger.info('\t'*2 + 'TEST last10mean loss {:.5f}'.format(test_loss))
        if test_loss < best_test_loss:
            logger.info('save optimizer test_loss {:.4f} -> {:.4f}'
                        .format(best_test_loss, test_loss))
            visualize(np.stack([test_all_losses, adam_test_all_losses], axis=2),
                      name=optimizer_path + '.testloss.png')
            chainer.serializers.save_npz(optimizer_path, optimizer.optnet)
            best_test_loss = test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='if cpu, use -1')
    parser.add_argument('--out', '-o', default='result')

    parser.add_argument('--cycle-meta', type=int, default=200)  # unk
    parser.add_argument('--evaluation-runs', type=int, default=5)
    parser.add_argument('--iter-meta', type=int, default=100)
    parser.add_argument('--unroll-iters', type=int, default=20)

    parser.add_argument('--adam-alpha', type=float, default=0.03)
    parser.add_argument('--out-scale', type=float, default=0.1)
    parser.add_argument('--optimizer-path', type=str,
                        default='optnet_seed{seed}.npz')

    parser.add_argument('--seed', type=int, default=7772)
    parser.add_argument('--use-adam', action='store_true', default=False)
    parser.add_argument('--do-pretraining', action='store_true', default=False)
    args = parser.parse_args()

    if '{seed}' in args.optimizer_path:
        changed = args.optimizer_path.replace('{seed}', str(args.seed))
        logger.info('optimizer_path {} -> {}'
                    .format(args.optimizer_path, changed))
        args.optimizer_path = changed
    train_optimizer(args)


if __name__ == '__main__':
    main()
