import chainer
import chainer.functions as F


def nll_gaussian(preds, target, variance, add_const=False):
    """ Calc a negative log likelihood with gaussian distribution.

    Args:
        preds (numpy.ndarray or cupy.ndarray):

    """
    if add_const:
        xp = backends.cuda.get_array_module(target)
        if isinstance(variance, chainer.Variable):
            variance = variance.array
        variance = xp.log(variance, dtype=target.dtype)
        F.gaussian_nll(preds, target, variance)
        neg_log_p = ((preds - target) ** 2 / (2 * variance))
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

