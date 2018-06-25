import math

import chainer
from chainer import cuda
import numpy


def warmup_cosine(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return s * (x / warmup) + (1 - s) * (0.5 * (1 + math.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return s * (x / warmup) + (1 - s) * 1


def warmup_linear(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return (s * (x / warmup) + (1 - s)) * (1 - x)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


def _scheduled_learning_rate(schedule, hp, t):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Adam optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    lrt = hp.alpha * math.sqrt(fix2) / fix1
    lrt *= schedule(t / hp.t_total, hp.warmup)
    return lrt


class OpenAIAdamRule(chainer.optimizers.adam.AdamRule):
    """Implements Open AI version of Adam algorithm with weight decay fix.

    The used adam has some differences from normal adam,
    although most of lines are same as https://github.com/chainer/chainer/blob/v4.1.0/chainer/optimizers/adam.py.
    """

    def __init__(self, schedule, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, eps=None,
                 eta=None, weight_decay_rate=None, amsgrad=None):
        super(OpenAIAdamRule, self).__init__(
            parent_hyperparam,
            alpha, beta1, beta2, eps,
            eta, weight_decay_rate, amsgrad)
        self.schedule = schedule
        assert not amsgrad  # amsgrad implementation is skipped

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        m, v = self.state['m'], self.state['v']

        m += (1 - hp.beta1) * (grad - m)
        v += (1 - hp.beta2) * (grad * grad - v)

        vhat = v
        # This adam multipies schduled adaptive learning rate
        # with both main term and weight decay.
        # Normal Adam: param.data -= hp.eta * (self.lr * m / (numpy.sqrt(vhat) + hp.eps) +
        #                                      hp.weight_decay_rate * param.data)
        param.data -= hp.eta * self.lr * (m / (numpy.sqrt(vhat) + hp.eps) +
                                          hp.weight_decay_rate * param.data)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))

        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, \
             T eta, T weight_decay_rate',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= eta * lr * (m / (sqrt(v) + eps) +
                               weight_decay_rate * param);''',
            'adam')(grad, self.lr, 1 - hp.beta1,
                    1 - hp.beta2, hp.eps,
                    hp.eta, hp.weight_decay_rate,
                    param.data, self.state['m'], self.state['v'])

    @property
    def lr(self):
        return _scheduled_learning_rate(self.schedule, self.hyperparam, self.t)


class OpenAIAdam(chainer.optimizers.Adam):
    def __init__(self,
                 warmup,
                 t_total,
                 schedule,
                 alpha=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 eta=1.0,
                 weight_decay_rate=0.,
                 amsgrad=False):
        super(OpenAIAdam, self).__init__(
            alpha, beta1, beta2, eps,
            eta, weight_decay_rate, amsgrad)
        assert t_total is not None
        self.hyperparam.warmup = warmup
        self.hyperparam.t_total = t_total * 1.0

        self.warmup = chainer.optimizer.HyperparameterProxy('warmup')
        self.t_total = chainer.optimizer.HyperparameterProxy('t_total')
        self.schedule = schedule

    def create_update_rule(self):
        return OpenAIAdamRule(self.schedule, self.hyperparam)


def get_OpenAIAdam(models,
                   lr, schedule,
                   warmup, t_total, b1=0.9,
                   b2=0.999, e=1e-8, l2=0., vector_l2=False,
                   max_grad_norm=-1):
    opt = OpenAIAdam(schedule=SCHEDULES[schedule],
                     warmup=warmup, t_total=t_total,
                     alpha=lr, beta1=b1,
                     beta2=b2, eps=e, weight_decay_rate=l2)
    opt.setup(chainer.ChainList(*models))  # tricky
    # TODO: grad norm clipping
    # TODO: vector l2
    return opt
