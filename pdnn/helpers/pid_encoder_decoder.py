import numpy as np
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import bad_value
from scipy import signal
from scipy.signal import butter, cheby1, lfilter_zi, lfilter


class LowpassFilter(object):

    def __init__(self, threshold_freq, order=2, design='cheby1'):
        """
        :param threshold_freq: Threshold frequency to filter out, as a fraction of sampling freq.  E.g. 0.1
        """
        self.b, self.a = \
            butter(N=order, Wn=threshold_freq, btype='low') if design == 'butter' else \
            cheby1(N=order, rp=0.1, Wn=threshold_freq, btype='low') if design == 'cheby1' else \
            bad_value(design)
        self.filter_state = lfilter_zi(b=self.b, a=self.a)

    def __call__(self, x):
        y, self.filter_state = lfilter(b=self.b, a=self.a, x = np.array(x)[None], zi = self.filter_state)
        return y[0]


class Herder(object):

    def __init__(self):
        self.phi = 0.

    def __call__(self, x):
        phi_prime = self.phi + x
        s = np.round(phi_prime)
        self.phi = phi_prime - s
        return s

    def reset(self):
        self.phi = 0.


class PIDDecoder(object):

    def __init__(self, kp, ki=0., kd=0.):
        self.kd = kd
        self.ki = ki
        self.one_over_kpid = 1./float(kp + ki + kd) if (kp + ki + kd)>0 else np.inf
        # self.kd_minus_ki = kd-ki
        self.xp = 0.
        self.sp = 0.

    def __call__(self, y):
        x = self.one_over_kpid * (y - self.ki*self.sp + self.kd*self.xp)
        self.sp += x
        self.xp = x
        return x


class PIDEncoder(object):

    def __init__(self, kp, ki=0., kd=0., noise = 0., rng = None):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.xp = 0
        self.s = 0
        self.noise = noise
        self.rng = get_rng(rng)

    def __call__(self, x):
        self.s += x
        y = self.kp*x + self.ki*self.s + self.kd*(x-self.xp)
        self.xp = x.copy()
        if self.noise!=0:
            y += self.rng.randn(*y.shape)*self.noise
        return y


def lowpass(sig, cutoff):
    b, a = butter(N=2, Wn=cutoff)
    new_sig = signal.lfilter(b, a, sig, axis=0)
    return new_sig


def lowpass_random(n_samples, cutoff, n_dim=None, rng = None, normalize = False, slope=0):
    """
    Return a random lowpass-filtered signal.
    :param n_samples:
    :param cutoff:
    :param rng:
    :return:
    """
    rng = get_rng(rng)
    assert 0<=cutoff<=1, "Cutoff must be in the range 0 (pure DC) to 1 (sample frequency)"
    base_signal = rng.randn(n_samples) if n_dim is None else rng.randn(n_samples, n_dim)
    lowpass_signal = lowpass(base_signal, cutoff)
    if normalize:
        lowpass_signal = lowpass_signal/np.std(lowpass_signal)
    if slope != 0:
        ramp = slope*np.arange(len(lowpass_signal))
        lowpass_signal = lowpass_signal+(ramp if n_dim is None else ramp[:, None])
    return lowpass_signal


def pid_encode(x, kp, ki=0., kd=0., quantization=None):

    enc = PIDEncoder(kp=kp, ki=ki, kd=kd)
    pid_encoded = np.array([enc(xi) for xi in x])
    if quantization is None:
        return pid_encoded
    elif quantization== 'herd':
        quantizer = Herder()
        quantized = np.array([quantizer(xi) for xi in pid_encoded])
        return quantized
    else:
        raise Exception('No quantizer: {}'.format(quantization))


def pid_decode(x, kp, ki=0., kd=0.):
    dec = PIDDecoder(kp=kp, ki=ki, kd=kd)
    decoded = np.array([dec(xi) for xi in x])
    return decoded
