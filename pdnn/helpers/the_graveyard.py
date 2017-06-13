import numpy as np
from plato.core import as_floatx, create_shared_variable, symbolic, add_update
from theano import tensor as tt


class FutureWeightGradCalculator(object):

    def __init__(self, kp, kd, shapes):
        """
        :param kp:
        :param kd:
        :param shapes: A tuple that specifies (minibatch_size, n_in, n_out)
        """
        self.kp = kp
        self.kd = kd
        self.r = kd/as_floatx(kp+kd)
        self.scale = (1./as_floatx(kp**2 + 2*kp*kd))
        self.x_past = create_shared_variable(np.zeros((shapes[0], shapes[1])))
        self.e_past = create_shared_variable(np.zeros((shapes[0], shapes[2])))

    @symbolic
    def compute_grad(self, xc, ec, x_true = None, e_true = None):
        """
        :param xc:
        :param ec:
        :param x:
        :param e:
        :return:
        """
        x_past = self.x_past*self.r if x_true is None else x_true*(self.kp+self.kd)-xc
        e_past = self.e_past*self.r if e_true is None else e_true*(self.kp+self.kd)-ec
        w_grad = self.scale * (xc.T.dot(e_past+ec) + x_past.T.dot(ec))
        if x_true is None:
            add_update(self.x_past, x_past + xc)
        if e_true is None:
            add_update(self.e_past, e_past + ec)
        return w_grad


@symbolic
def past_weight_grad_calculator2(xs, es, kp_x, kd_x, kp_e, kd_e, shapes):
    """
    This attempt never really got off the ground.  It doesn't work
    """
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    n_samples, n_in, n_out = shapes
    rx = kd_x/(kp_x+kd_x)
    re = kd_e/(kp_e+kd_e)

    xr = create_shared_variable(np.zeros((n_samples, n_in)))
    er = create_shared_variable(np.zeros((n_samples, n_out)))




    # xr_new = xr*rx + xs/(kp_x+kd_x)
    # er_new = er*re + es/(kp_e+kd_e)

    arr = rx*re/(1-rx*re)

    xr_new = xr*arr + xs/(kp_x+kd_x)
    er_new = er*arr + es/(kp_e+kd_e)

    xsum = create_shared_variable(np.zeros((n_samples, n_in)))
    esum = create_shared_variable(np.zeros((n_samples, n_out)))

    xsum_new = xsum+xr_new
    esum_new = esum+er_new

    x_nospikes = tt.eq(xs, 0)
    e_nospikes = tt.eq(es, 0)

    dw = xs.T.dot(esum_new) + xsum_new.T.dot(es)

    add_update(xr, xr_new)
    add_update(er, er_new)
    add_update(xsum, xsum_new*x_nospikes)
    add_update(esum, esum_new*e_nospikes)

    return xs.T.dot(er) + xr.T.dot(es)
    # return xr.T.dot(er)
    # return dw