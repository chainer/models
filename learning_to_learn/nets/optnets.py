import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class OptimizerByNet(object):
    def __init__(self, optnet, grand_optmizer):
        self.grand_optimizer = grand_optmizer
        self.grand_optimizer.setup(optnet)
        self.optnet = optnet
        self.optnet.cleargrads()

        self.targets = []
        self.weakref_cache = []

    def setup(self, *targets):
        self.release_all()
        self.targets = targets

    def trancate_params(self):
        for target in self.targets:
            for name, param in sorted(target.namedparams()):
                param.unchain()

    def reset_cache_for_weakref(self):
        self.weakref_cache = []

    def release_all(self):
        self.reset_cache_for_weakref()
        self.targets = []
        self.optnet.reset_state()
        self.optnet.cleargrads()

    def set_param(self, link, name, value, train_optnet=True):
        value.name = name
        if not train_optnet:
            value.unchain()
        super(chainer.Link, link).__setattr__(name, value)

    def meta_update(self):
        self.grand_optimizer.update()
        self.reset_cache_for_weakref()
        self.trancate_params()
        self.optnet.trancate_state()

    def update(self, train_optnet=True):
        # calculate
        sorted_namedparams = []
        sorted_grads = []
        for target in self.targets:
            for name, param in sorted(target.namedparams()):
                sorted_grads.append(param.grad)
        concat_grads = F.concat(
            [grad.reshape(-1) for grad in sorted_grads], axis=0).array[:, None]
        concat_gs = self.optnet.step(concat_grads)

        if not train_optnet:
            self.optnet.trancate_state()
            self.optnet.cleargrads()

        # update
        read_size = 0
        for target in self.targets:
            name2link = dict(target.namedlinks())
            for name, param in sorted(target.namedparams()):
                if train_optnet:
                    self.weakref_cache.append(param)  # no need?
                # update
                split_idx = name.rindex('/')
                link_name, attr_name = name[:split_idx], name[split_idx + 1:]
                g = concat_gs[read_size:read_size + param.size].\
                    reshape(param.shape)
                read_size += param.size
                self.set_param(name2link[link_name], attr_name, param + g,
                               train_optnet=train_optnet)


def preprocess_grad(x, p=10.):
    # pre-processing according to Deepmind 'Learning to Learn' paper
    xp = chainer.cuda.get_array_module(x)
    threshold = xp.exp(-p)

    x_abs = xp.abs(x)
    is_higher = x_abs >= threshold

    processed_x = [xp.zeros_like(x), xp.zeros_like(x)]
    processed_x[0][is_higher] = xp.log(x_abs[is_higher] + 1e-8) / p
    processed_x[1][is_higher] = xp.sign(x[is_higher])
    processed_x[0][~is_higher] = -1.
    processed_x[1][~is_higher] = x[~is_higher] * xp.exp(p)
    processed_x = xp.concatenate(processed_x, axis=1)

    return processed_x


class LSTMOptNet(chainer.Chain):
    def __init__(self, n_units=20, n_classes=10, out_scale=0.1,
                 do_preprocess=True):
        super(LSTMOptNet, self).__init__()
        with self.init_scope():
            n_input = 2 if do_preprocess else 1
            self.l1 = L.LSTM(n_input, 20, forget_bias_init=0)
            self.l2 = L.LSTM(20, 20, forget_bias_init=0)
            self.lout = L.Linear(20, 1)
            # self.ldirect = L.Linear(n_input, 1)
        self.do_preprocess = do_preprocess
        self.out_scale = out_scale

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def trancate_state(self):
        def unchain(state):
            if state is not None:
                state.unchain()
        unchain(self.l1.c)
        unchain(self.l1.h)
        unchain(self.l2.c)
        unchain(self.l2.h)

    def step(self, x):
        if self.do_preprocess:
            x = preprocess_grad(x)
        h1 = self.l1(x)
        h2 = self.l2(h1)  # + h1
        g = self.lout(h2)  # + self.ldirect(x)
        return g * self.out_scale

    def __call__(self, *args):
        raise NotImplementedError
