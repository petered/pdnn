from abc import abstractmethod
import numpy as np
import theano
from artemis.general.should_be_builtins import izip_equal
from artemis.ml.tools.neuralnets import initialize_network_params
from plato.core import symbolic, create_shared_variable, create_constant, add_update, as_theano_variable, as_floatx
from plato.interfaces.helpers import compute_activation, shared_like
from plato.tools.common.online_predictors import ISymbolicPredictor
from plato.tools.optimization.cost import get_named_cost_function
from theano import tensor as tt
from theano.ifelse import ifelse
from pdnn.helpers.the_graveyard import FutureWeightGradCalculator


class PDHerdingNetwork(ISymbolicPredictor):

    def __init__(self, ws, hidden_activation, output_activation, optimizer, loss, encdec, encdec_back = None,
             grad_calc='xx', minibatch_size=1):
        """
        :param ws:
        :param encdec: A PDEncoderDecoder pair
        :param hidden_activation:
        :param output_activation:
        :param optimizer:
        :param loss:
        :param enddec_back: Opt
        :param grad_calc:
        :param minibatch_size:
        :param fwd_quantizer:
        :param back_quantizer:
        """

        if isinstance(encdec, dict):
            encdec = PDEncoderDecoder(kp=encdec['kp'], kd=encdec['kd'], quantization=encdec['quantizer'])
        if encdec_back is None:
            encdec_back = encdec
        elif isinstance(encdec_back, dict):
            encdec_back = PDEncoderDecoder(kp=encdec_back['kp'], kd=encdec_back['kd'], quantization=encdec_back['quantizer'])

        self.layers = [PDHerdingLayer(w, b=np.zeros(w.shape[1]), encdec=encdec if not callable(encdec) else encdec(), encdec_back=encdec_back if not callable(encdec_back) else encdec_back(),
                  nonlinearity=nonlinearity, grad_calc=grad_calc,  minibatch_size=minibatch_size)
                       for w, nonlinearity in izip_equal(ws, [hidden_activation]*(len(ws)-1)+[output_activation])]
        self.optimizer = optimizer
        self.loss = get_named_cost_function(loss) if isinstance(loss, basestring) else loss
        self.minibatch_size = minibatch_size

    @classmethod
    def from_init(cls, layer_sizes, initializer ='xavier', rng=None, **kwargs):
        ws = initialize_network_params(layer_sizes=layer_sizes, mag=initializer, include_biases=False, rng=rng)
        return PDHerdingNetwork(ws=ws, **kwargs)

    @symbolic
    def predict(self, x):
        outs = self.predict_one.scan(sequences = [x.reshape((-1, self.minibatch_size, x.shape[1]))])  # (n_minibatces, minibatch_size, out_size
        return outs.reshape((-1, outs.shape[2]))

    @symbolic
    def train(self, x, y):
        self.train_one.scan(
            sequences = (
                x.reshape((-1, self.minibatch_size, x.shape[1])),
                y.reshape((-1, self.minibatch_size, y.shape[1])) if y.ndim==2 else y.reshape((-1, self.minibatch_size)), )
                )

    @symbolic
    def predict_one(self, x):
        x = tt.unbroadcast(x, 0)  # F'ing scan
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    @symbolic
    def train_one(self, x, target):
        x, target = tt.unbroadcast(x, 0), tt.unbroadcast(target, 0)  # F'ing scan
        states = {}
        for layer in self.layers:
            x, layer_state = layer.forward_pass_and_state(x, count_ops=True)
            states[layer]=layer_state
        loss = self.loss(x, target)
        param_grad_pairs = []
        grad = None
        for layer in self.layers[::-1]:
            grad, param_grads = layer.backward_pass(state=states[layer], grad=grad, cost = loss, count_ops=True)
            loss = None
            param_grad_pairs += list(izip_equal(layer.parameters, param_grads))
        all_params, all_param_grads = zip(*param_grad_pairs)
        self.optimizer.update_from_gradients(parameters=all_params, gradients=all_param_grads)
        return create_constant(0.)  # scan demands some return

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]

    def get_op_count(self):
        return sum(layer.fwd_op_count.get_value() + layer.back_op_count.get_value() for layer in self.layers)


class PDHerdingLayer(object):

    def __init__(self, w, b, nonlinearity, encdec, encdec_back, grad_calc='xx', minibatch_size=1):
        self.n_in, self.n_out = w.shape
        self.w = create_shared_variable(w)
        self.b = create_shared_variable(b)

        assert isinstance(encdec, IEncoderDecoder)
        assert isinstance(encdec_back, IEncoderDecoder)
        self.encdec = encdec
        self.encdec_back = encdec_back
        self.nonlinearity = nonlinearity
        self.minibatch_size = minibatch_size
        self.grad_calc = grad_calc
        self.fwd_op_count = create_shared_variable(0, name='fwd_op_count')
        self.back_op_count = create_shared_variable(0, name='back_op_count')
        self.update_op_count = create_shared_variable(0, name='update_op_count')

    def forward_pass_and_state(self, x, count_ops = False):
        # s = quantize(x, mode = self.fwd_quantizer, shape = (self.minibatch_size, self.n_in))
        # s = pd_encode(x, kp=self.kp, kd=self.kd, quantization=self.fwd_quantizer, shape = (self.minibatch_size, self.n_in))
        s = self.encdec.encode(x, shape = (self.minibatch_size, self.n_in))

        # if self.n_in==784:
        #     tdbplot(s.reshape((28, 28)), 's')

        # pre_act = pd_decode(s.dot(self.w), kp=self.kp, kd=self.kd, shape= (self.minibatch_size, self.n_out)) + self.b
        pre_act = self.encdec.decode(s.dot(self.w), shape= (self.minibatch_size, self.n_out)) + self.b

        if count_ops:
            add_update(self.fwd_op_count, self.fwd_op_count+abs(s).sum().astype('int64')*self.n_out, accumulate=True)

        # pre_act = s.dot(self.w) + self.b
        out = compute_activation(pre_act, activation_name=self.nonlinearity)
        return out, (x, s, pre_act)

    def forward_pass(self, x):
        """
        :param x: A real (n_samples, n_dims_in) input
        :return: A real (n_samples, n_dims_in) output
        """
        out, _ = self.forward_pass_and_state(x)
        return out

    def backward_pass(self, state, grad, cost = None, count_ops = False):
        """
        :param grad: An integer (n_samples, n_dim_out) gradient estimate
        :return: (delta, param_gradients) Where:
            delta: A (n_samples, n_dim_in) integer gradient estimate
        """

        assert (grad is None) != (cost is None), "You can either pass grad or cost"

        ap, ap_q, z = state


        if cost is None:
            filters = tt.grad(compute_activation(z, activation_name=self.nonlinearity).sum(), wrt=z)
            grad_z = filters*grad
        elif grad is None:
            grad_z = tt.grad(cost, wrt=z)

        # sb = quantize(pre_act_grad, mode=self.back_quantizer, shape=(self.minibatch_size, self.n_out))
        # grad_z_q = pd_encode(grad_z, kp=self.kp_back, kd=self.kd_back, quantization=self.back_quantizer, shape=(self.minibatch_size, self.n_out))
        grad_z_q = self.encdec_back.encode(grad_z, shape=(self.minibatch_size, self.n_out))

        if count_ops:
            add_update(self.back_op_count, self.back_op_count+abs(grad_z_q).sum().astype('int64')*self.n_in, accumulate=True)

        # grad_ap = pd_decode(grad_z_q.dot(self.w.T), kp=self.kp_back, kd=self.kd_back, shape=(self.minibatch_size, self.n_in))
        grad_ap = self.encdec_back.decode(grad_z_q.dot(self.w.T), shape=(self.minibatch_size, self.n_in))

        if self.grad_calc in ('true', 'xx', 'recon'):  # Dense op count
            add_update(self.update_op_count, self.back_op_count+self.minibatch_size*self.n_in*self.n_out)
        elif self.grad_calc in ('future', 'future-true', 'past', 'past_step', 'past_reloaded', 'past_matrix'):  # Sparse op count
            add_update(self.update_op_count, self.update_op_count+abs(ap_q).sum().astype('int64')*self.n_out + abs(grad_z_q).sum().astype('int64')*self.n_in)
        else:
            raise NotImplementedError('No op-count method for {}'.format(self.grad_calc))

        w_grad = self._get_past_gradient(ap, grad_z, ap_q, grad_z_q, grad_calc=self.grad_calc)
        # tdbplot(w_grad, self.grad_calc)
        # w_reloaded = self._get_past_gradient(ap, grad_z, ap_q, grad_z_q, grad_calc='past_reloaded')
        # tdbplot(w_reloaded, 'reloaded')

        b_grad = grad_z_q.sum(axis=0) if self.grad_calc[-1]=='s' else grad_z.sum(axis=0)
        return grad_ap, [w_grad, b_grad]

    @property
    def parameters(self):
        return [self.w, self.b]

    def _get_past_gradient(self, ap, grad_z, ap_q, grad_z_q, grad_calc=None):

        if grad_calc in ('true', 'xx'):
            w_grad = ap.T.dot(grad_z)
        elif grad_calc=='recon':
            ap_r = self.encdec.decode(ap_q, shape=(self.minibatch_size, self.n_in))
            grad_z_r = self.encdec_back.decode(grad_z_q, shape=(self.minibatch_size, self.n_out))
            w_grad = ap_r.T.dot(grad_z_r)
        elif grad_calc=='future':
            w_grad = future_weight_grad_calculator(xs=ap_q, es=grad_z_q, kp_x=self.encdec.kp, kd_x=self.encdec.kd, kp_e=self.encdec_back.kp, kd_e=self.encdec_back.kd, shapes=(self.minibatch_size, self.n_in, self.n_out))
        elif grad_calc=='future-true':
            assert self.encdec.kp.ivalue == self.encdec_back.kp.ivalue
            assert self.encdec.kd.ivalue == self.encdec_back.kd.ivalue
            w_grad = FutureWeightGradCalculator(kp=self.kp, kd=self.kd, shapes=(self.minibatch_size, self.n_in, self.n_out)).compute_grad(xc=ap_q, ec=grad_z_q, x_true=ap, e_true=grad_z)
        elif grad_calc=='past':
            w_grad = past_weight_grad_calculator(xs=ap_q, es=grad_z_q, kp_x=self.encdec.kp, kd_x=self.encdec.kd, kp_e=self.encdec_back.kp, kd_e=self.encdec_back.kd, shapes=(self.minibatch_size, self.n_in, self.n_out))
        elif grad_calc=='past_reloaded':
            w_grad = past_weight_grad_calculator_reloaded(xs=ap_q, es=grad_z_q, kp_x=self.encdec.kp, kd_x=self.encdec.kd, kp_e=self.encdec_back.kp, kd_e=self.encdec_back.kd, shapes=(self.minibatch_size, self.n_in, self.n_out))
        elif grad_calc=='past_matrix':
            w_grad = matrix_weight_grad_calculator(xs=ap_q, es=grad_z_q, kp_x=self.encdec.kp, kd_x=self.encdec.kd, kp_e=self.encdec_back.kp, kd_e=self.encdec_back.kd, shapes=(self.minibatch_size, self.n_in, self.n_out))
        elif grad_calc=='past_step':
            assert self.minibatch_size==1, "This ain't gonna work."
            w_grad = past_weight_grad_step(xs=ap_q[0], es=grad_z_q[0], kp_x=self.encdec.kp, kd_x=self.encdec.kd, kp_e=self.encdec_back.kp, kd_e=self.encdec_back.kd, shape=(self.n_in, self.n_out))
        else:
            raise NotImplementedError(grad_calc)
        return w_grad


class IEncoderDecoder(object):

    @abstractmethod
    def encode(self, x, shape=None):
        pass

    @abstractmethod
    def decode(self, x, shape=None):
        pass


class PDAdaptiveEncoderDecoder(IEncoderDecoder):

    def __init__(self, kp, kd, adaptation_rate = 0.0001, quantization = None):
        """

        :param kp_over_kd: The ratio of kp/kd.  0.01 might be a normal value.
        :param relative_scale: Try to maintain a scale of
        :param adaptation_rate:
        """

        self.k_alpha = kd/float(kp+kd)
        self.k_beta_init = 1/float(kp+kd)  # The scale
        self.k_beta=self.k_beta_init
        assert np.allclose(self.kp, kp)
        assert np.allclose(self.kd, kd)
        self.k_beta = create_shared_variable(self.k_beta_init)
        self.adaptation_rate = adaptation_rate
        self.quantization = quantization

    @property
    def kp(self):
        return (1-self.k_alpha)/self.k_beta

    @property
    def kd(self):
        return self.k_alpha/self.k_beta

    @symbolic
    def encode(self, x, shape=None):
        running_mag = create_shared_variable(1.)
        add_update(running_mag, (1-self.adaptation_rate)*running_mag + self.adaptation_rate*abs(x).mean())
        target_k_beta = self.k_beta_init*running_mag
        add_update(self.k_beta, self.k_beta + self.adaptation_rate*(target_k_beta - self.k_beta))
        return pd_encode(x, kp=self.kp, kd=self.kd, quantization=self.quantization, shape=shape)

    @symbolic
    def decode(self, x, shape=None):
        return pd_decode(x, kp=self.kp, kd=self.kd, shape=shape)


@symbolic
def pd_encode(x, kp, kd, quantization=None, shape=None):
    # TODO: Should probably put the meat in here and call it externally
    return PDEncoderDecoder(kp=kp, kd=kd, quantization=quantization).encode(x, shape=shape)


@symbolic
def pd_decode(y, kp, kd, shape=None):
    # TODO: Should probably put the meat in here and call it externally
    return PDEncoderDecoder(kp=kp, kd=kd).decode(y, shape=shape)


class PDEncoderDecoder(IEncoderDecoder):

    def __init__(self, kp, kd, quantization = None):
        self.kp = as_theano_variable(kp, dtype='floatX')
        self.kd = as_theano_variable(kd, dtype='floatX')
        self.quantization = quantization

    def encode(self, x, shape=None):
        if shape is None:
            xp = create_shared_variable(np.zeros((0, )*x.ndim), name='xp')
            delta = ifelse(xp.size>0, x-xp, x)
        else:
            xp = create_shared_variable(np.zeros(shape), name='xp{}'.format(shape))
            delta = x - xp
        add_update(xp, x)
        y = self.kp*x + self.kd*delta
        if self.quantization is None:
            return y
        elif self.quantization=='herd':
            return herd(y, shape=shape)
        else:
            raise Exception('No quantizer: {}'.format(self.quantization))

    def decode(self, y, shape=None):
        xp = shared_like(y, name='xp') if shape is None else create_shared_variable(np.zeros(shape), name='xp{}'.format(shape))
        div = (self.kp+self.kd)
        x = (y+self.kd*xp)/div
        add_update(xp, x)
        return x


@symbolic
def herd(x, shape = None):
    phi = shared_like(x, name='phi') if shape is None else create_shared_variable(np.zeros(shape), name='phi{}'.format(shape))
    phi_ = phi + x
    s = tt.round(phi_)
    add_update(phi, phi_ - s)
    return s


@symbolic
def past_weight_grad_calculator(xs, es, kp_x, kd_x, kp_e, kd_e, shapes):
    """
    Do an efficient update of the weights given the two spike-trains.

    This isn't actually implemented as an efficient update, but it will produce the identical result as if it were.

    :param xs: An (n_samples, n_in) array
    :param es: An (n_samples, n_out) array
    :param kp_x: kp for the x units
    :param kd_x: kd for the x units
    :param kp_e: kp for the e units
    :param kd_e: kd for the e units
    :param shapes: (minibatch_size, n_in, n_out)
    :return: An (n_in, n_out) approximate weight gradient.
    """
    # TODO: Make this actually use sparsity, one of these days.
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    n_samples, n_in, n_out = shapes
    rx = kd_x/(kp_x+kd_x)
    re = kd_e/(kp_e+kd_e)

    tx_last = create_shared_variable(np.zeros((n_samples, n_in))+1)
    te_last = create_shared_variable(np.zeros((n_samples, n_out))+1)
    x_last = create_shared_variable(np.zeros((n_samples, n_in)))
    e_last = create_shared_variable(np.zeros((n_samples, n_out)))

    t_last = tt.minimum(tx_last[:, :, None], te_last[:, None, :])
    x_spikes = tt.neq(xs, 0)
    dw_potentials = x_last[:, :, None] * e_last[:, None, :] * \
            rx**(tx_last[:, :, None]-t_last) \
            * re**(te_last[:, None, :]-t_last) \
            * geoseries_sum(rx*re, t_end=t_last, t_start=1)
    e_spikes = tt.neq(es, 0)
    dws = (x_spikes[:, :, None]+e_spikes[:, None, :]-x_spikes[:, :, None]*e_spikes[:, None, :])*dw_potentials  # (n_samples, n_in, n_out)

    add_update(x_last, tt.switch(x_spikes, x_last*rx**tx_last + xs/as_floatx(kd_x), x_last))
    add_update(e_last, tt.switch(e_spikes, e_last*rx**te_last + es/as_floatx(kd_e), e_last))
    add_update(tx_last, tt.switch(x_spikes, 1, tx_last+1))
    add_update(te_last, tt.switch(e_spikes, 1, te_last+1))
    return dws.sum(axis=0)



@symbolic
def past_weight_grad_calculator_reloaded(xs, es, kp_x, kd_x, kp_e, kd_e, shapes):
    """
    Do an efficient update of the weights given the two spike-trains.

    This isn't actually implemented as an efficient update, but it will produce the identical result as if it were.

    :param xs: An (n_samples, n_in) array
    :param es: An (n_samples, n_out) array
    :param kp_x: kp for the x units
    :param kd_x: kd for the x units
    :param kp_e: kp for the e units
    :param kd_e: kd for the e units
    :param shapes: (minibatch_size, n_in, n_out)
    :return: An (n_in, n_out) approximate weight gradient.
    """
    # TODO: RESOLVE INSTABILITY ISSUE
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    n_samples, n_in, n_out = shapes
    rx = kd_x/(kp_x+kd_x)
    re = kd_e/(kp_e+kd_e)

    tx_last = create_shared_variable(np.zeros((n_samples, n_in)))
    te_last = create_shared_variable(np.zeros((n_samples, n_out)))
    xr = create_shared_variable(np.zeros((n_samples, n_in)))
    er = create_shared_variable(np.zeros((n_samples, n_out)))

    x_spikes = tt.neq(xs, 0)
    e_spikes = tt.neq(es, 0)
    t_last = tt.maximum(tx_last[:, :, None], te_last[:, None, :])
    sum_to_last = geoseries_sum(rx*re, t_start=t_last, t_end=0)  # Wasteful, since most of this is multiplied by zeros later, but for now it don't matter

    spikes = tt.bitwise_or(x_spikes[:, :, None], e_spikes[:, None, :])
    dw_es = (xr[:, :, None]*er[:, None, :]*spikes)*sum_to_last  # PROBLEM HERE!!!! Can be very small number times very large numen
    # dw_es = (xr[:, :, None]*(x_spikes[:, :, None]-x_spikes[:, :, None]*e_spikes[:, None, :]) * er[:, None, :] + xr[:, :, None] * (er*e_spikes)[:, None, :]) * sum_to_last
    # dw_es = (xr[:, :, None]*(x_spikes[:, :, None]-x_spikes[:, :, None]*e_spikes[:, None, :]) * er[:, None, :] + xr[:, :, None] * (er*e_spikes)[:, None, :]) * sum_to_last
    add_update(xr, xr*rx + xs/(kp_x+kd_x))
    add_update(er, er*re + es/(kp_e+kd_e))
    add_update(tx_last, tt.switch(x_spikes, 0, tx_last-1))
    add_update(te_last, tt.switch(e_spikes, 0, te_last-1))

    return dw_es.sum(axis=0)


@symbolic
def matrix_weight_grad_calculator(xs, es, kp_x, kd_x, kp_e, kd_e, shapes, epsilon=1e-7):
    """
    :param xs:
    :param es:
    :param kp_x:
    :param kd_x:
    :param kp_e:
    :param kd_e:
    :param shapes:
    :param epsilon:
    :return:
    """
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    n_samples, n_in, n_out = shapes
    v1 = create_shared_variable(np.zeros((n_samples, n_in, n_out)))
    rx = kd_x/(kp_x+kd_x)
    re = kd_e/(kp_e+kd_e)
    xr = create_shared_variable(np.zeros((n_samples, n_in)))
    er = create_shared_variable(np.zeros((n_samples, n_out)))

    x_spikes = tt.neq(xs, 0)
    e_spikes = tt.neq(es, 0)
    xr_decayed = xr*rx
    er_decayed = er*re
    spikes = tt.bitwise_or(x_spikes[:, :, None], e_spikes[:, None, :])
    v2 = xr_decayed[:, :, None]*er_decayed[:, None, :]
    dws = (spikes*(v2-v1))/(rx*re-1)
    new_xr = xr_decayed + xs/(kp_x+kd_x)
    new_er = er_decayed + es/(kp_e+kd_e)

    add_update(v1, tt.switch(spikes, new_xr[:, :, None]*new_er[:, None, :], v1))
    add_update(xr, new_xr)
    add_update(er, new_er)

    return dws.sum(axis=0)


@symbolic
def future_weight_grad_calculator(xs, es, kp_x, kd_x, kp_e, kd_e, shapes):
    """
    Do an efficient update of the weights given the two spike-trains.

    This isn't actually implemented as an efficient update, but it will produce the identical result as if it were.

    :param xs: An (n_samples, n_in) array
    :param es: An (n_samples, n_out) array
    :param kp_x: kp for the x units
    :param kd_x: kd for the x units
    :param kp_e: kp for the e units
    :param kd_e: kd for the e units
    :param shapes: (minibatch_size, n_in, n_out)
    :return: An (n_in, n_out) approximate weight gradient.
    """
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    rx = kd_x/as_floatx(kp_x+kd_x)
    re = kd_e/as_floatx(kp_e+kd_e)
    scale = (1./as_floatx(kp_x*kp_e + kp_x*kd_e + kd_x*kp_e))
    n_samples, n_in, n_out = shapes
    x_past_var = create_shared_variable(np.zeros((n_samples, n_in)))
    e_past_var = create_shared_variable(np.zeros((n_samples, n_out)))
    x_past = x_past_var*rx
    e_past = e_past_var*re
    w_grad = scale * (xs.T.dot(e_past+es) + x_past.T.dot(es))
    add_update(x_past_var, x_past + xs)
    add_update(e_past_var, e_past + es)
    return w_grad


@symbolic
def past_weight_grad_step(xs, es, kp_x, kd_x, kp_e, kd_e, shape, dws=None):
    """
    Do an efficient update of the weights given the two spike-update.

    (This still runs FING SLOWLY!)

    :param xs: An (n_in) vector
    :param es: An (n_out) vector
    :param kp_x:
    :param kd_x:
    :param kp_e:
    :param kd_e:
    :param shapes: (n_in, n_out)
    :return:
    """
    kp_x, kd_x, kp_e, kd_e = [as_floatx(k) for k in (kp_x, kd_x, kp_e, kd_e)]
    n_in, n_out = shape
    rx = kd_x/(kp_x+kd_x)
    re = kd_e/(kp_e+kd_e)

    tx_last = create_shared_variable(np.zeros(n_in)+1)
    te_last = create_shared_variable(np.zeros(n_out)+1)
    x_last = create_shared_variable(np.zeros(n_in))
    e_last = create_shared_variable(np.zeros(n_out))
    x_spikes = tt.neq(xs, 0)
    e_spikes = tt.neq(es, 0)
    x_spike_ixs, = tt.nonzero(x_spikes)
    e_spike_ixs, = tt.nonzero(e_spikes)

    if dws is None:
        dws = tt.zeros(shape)

    t_last = tt.minimum(tx_last[x_spike_ixs, None], te_last)  # (n_x_spikes, n_out)
    dws = tt.inc_subtensor(dws[x_spike_ixs, :], x_last[x_spike_ixs, None]*e_last
        * rx**(tx_last[x_spike_ixs, None]-t_last)
        * re**(te_last[None, :]-t_last)
        * geoseries_sum(re*rx, t_end=t_last, t_start=1)
        )

    new_x_last = tt.set_subtensor(x_last[x_spike_ixs], x_last[x_spike_ixs]*rx**tx_last[x_spike_ixs]+ xs[x_spike_ixs]/as_floatx(kd_x))
    new_tx_last = tt.switch(x_spikes, 0, tx_last)

    t_last = tt.minimum(new_tx_last[:, None], te_last[e_spike_ixs])  # (n_in, n_e_spikes)
    dws = tt.inc_subtensor(dws[:, e_spike_ixs], new_x_last[:, None]*e_last[e_spike_ixs]
        * rx**(new_tx_last[:, None]-t_last)
        * re**(te_last[None, e_spike_ixs]-t_last)
        * geoseries_sum(re*rx, t_end=t_last, t_start=1)
        )

    add_update(x_last, new_x_last)
    add_update(e_last, tt.set_subtensor(e_last[e_spike_ixs], e_last[e_spike_ixs]*re**te_last[e_spike_ixs]+ es[e_spike_ixs]/as_floatx(kd_e)))
    add_update(tx_last, new_tx_last+1)
    add_update(te_last, tt.switch(e_spikes, 1, te_last+1))
    return dws


@symbolic
def geoseries_sum(r, t_end, t_start):
    """
    Sum of r**t from t=t_start to t=t_end, inclusive

    :param r:
    :param t_end:
    :param t_start:
    :return:
    """
    # return ifelse(tt.eq(r, 1), (t_end-t_start+1).astype(theano.config.floatX), (r**(t_end+1)-r**t_start)/(r-1))
    return ifelse(tt.eq(r, 1), (t_end-t_start+1).astype(theano.config.floatX), (r**(t_end+1)-r**t_start)/(r-1))


