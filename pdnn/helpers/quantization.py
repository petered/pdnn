from artemis.general.mymath import fixed_diff
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import bad_value
import numpy as np

__author__ = 'peter'


def discretize(x, approach='noisy-round', scale = 1, rng = None):

    if rng is None:
        rng = get_rng(rng)
    if approach == 'noisy-round':
        return np.round(x*scale + rng.uniform(low=-.5, high=.5, size=x.shape))/scale
    elif approach == 'round':
        return np.round(x*scale)/scale
    elif approach == 'noisy-add':
        return x + rng.uniform(-.5, .5, size=x.shape)/scale
    elif approach == 'surrogate-noise':
        return x + (12**.5)*((x%1)-(x%1)**2)*rng.uniform(low=-.5, high=.5, size=x.shape)/scale
    else:
        raise Exception('No discretization approach: %s' % approach)


def sequential_quantize(v, n_steps = None, method='herd', rng = None):
    """
    :param v: A (..., n_samples, n_units, ) array
    :param n_steps: The number of steps to spike for
    :return: An (..., n_steps, n_units) array of quantized values
    """
    rng = get_rng(rng)
    assert v.ndim>=2
    if n_steps is None:
        n_steps = v.shape[-2]
    else:
        assert n_steps == v.shape[-2]

    if method=='herd':
        result = fixed_diff(np.round(np.cumsum(v, axis=-2)), axis=-2)
    elif method=='herd2':
        result = fixed_diff(fixed_diff(np.round(np.cumsum(np.cumsum(v, axis=-2), axis=-2)), axis=-2), axis=-2)
    elif method=='round':
        result = np.round(v)
    elif method == 'slippery.9':
        result = slippery_round(v, slip=0.9)
    elif method == 'slippery.5':
        result = slippery_round(v, slip=0.5)
    elif method == 'randn':
        result = v + rng.randn(*v.shape)
    elif method=='uniform':
        result = v + rng.uniform(-.5, .5, size=v.shape)
    elif method=='surrogate-noise':
        result = v + (12**.5)*((v%1)-(v%1)**2)*rng.uniform(low=-.5, high=.5, size=v.shape)
    elif method == 'surrogate-sqrt':
        result = v + np.sqrt((12**.5)*((v%1)-(v%1)**2)*rng.uniform(low=-.5, high=.5, size=v.shape))
    elif method is None:
        result = v
    else:
        raise NotImplementedError("Don't have quantization method '%s' implemented" % (method, ))
    return result


def slippery_round(x, slip):
    """
    A soft sort of rounding, where we strike a balance between f(x)=x and f(x)=round(x).

    Critically, it is differentiable.

    :param x: An input
    :param slip: A real number between 0 and 1.  0 means identity function, 1 means rounding function.
    :return: (1-slip)*x + slip*round(x)
    """
    return (1-slip)*x + slip*np.round(x)


def quantize(v, method='herd', rng = None):
    """
    :param v: Can be:
        A (..., n_units) array
    :return:
        A (..., n_units) quantized array
    """
    v = v[..., None, :]
    quant = sequential_quantize(v, method=method, rng=rng)
    return quant[:, 0, :]


def quantize_sequence(v, method = 'herd', rng=None):
    """
    Do the rounding version of
    :param v: A (n_steps, n_dims) array of sequential inputs
    :return: A (n_steps, n_dims) integer array of activations.
    """
    return sequential_quantize(v[None], method=method, rng=rng)[0]
