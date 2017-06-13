import numpy as np
from artemis.general.mymath import geosum


def scalar_weight_updates(xc, ec, kp, kd):

    tx_last = 0
    te_last = 0
    x_last = 0
    e_last = 0
    n_samples = xc.shape[0]
    dw = np.zeros(n_samples)
    r = kd/float(kp+kd)

    for t in xrange(n_samples):

        t_last = max(tx_last, te_last)
        if xc[t]:
            dw[t] = x_last*e_last*r**(2*t_last-tx_last-te_last)*geosum(r**2, t_end=t-t_last+1, t_start=1)
            x_last = x_last * r**(t-tx_last) + xc[t]
            tx_last = t
        if ec[t]:
            dw[t] = x_last*e_last*r**(2*t_last-tx_last-te_last)*geosum(r**2, t_end=t-t_last+1, t_start=1)  # THing...
            e_last = e_last * r**(t-te_last) + ec[t]
            te_last = t
    return np.cumsum(dw)/kd**2


def pd_weight_grads(xc, ec, kp, kd):
    """
    Efficiently compute the weights over time for a system recieving sparse inputs xc and ec.

    :param xc: An (n_samples, n_in) array of input spikes
    :param ec:  An (n_samples, n_in) array of error_spikes
    :param kp: A scalar kp value
    :param kd: A scalar kd value
    :return: An (n_samples, n_in, n_out) array of weight updates
    """
    r = kd/float(kp+kd)
    n_samples = xc.shape[0]
    assert n_samples == ec.shape[0]
    n_in = xc.shape[1]
    n_out = ec.shape[1]
    dws = np.zeros((n_samples, n_in, n_out))
    tx_last = np.zeros(n_in)
    te_last = np.zeros(n_out)
    x_last = np.zeros(n_in)
    e_last = np.zeros(n_out)
    for t in xrange(n_samples):
        x_spikes = xc[t] != 0

        t_last = np.maximum(tx_last[x_spikes, None], te_last)
        dws[t, x_spikes, :] = x_last[x_spikes, None] * e_last * r**(2*t_last-tx_last[x_spikes, None]-te_last)*geosum(r**2, t_end=t-t_last, t_start=1)  # Not 100% on this...
        x_last[x_spikes] = x_last[x_spikes]*r**(t-tx_last[x_spikes]) + xc[t][x_spikes] / float(kd)
        tx_last[x_spikes] = t
        e_spikes = ec[t] != 0
        if np.any(e_spikes):
            t_last = np.maximum(tx_last[:, None], te_last[e_spikes])
            dws[t, :, e_spikes] += (x_last[:, None] * e_last[e_spikes] * r**(2*t_last-tx_last[:, None]-te_last[e_spikes])*geosum(r**2, t_end=t-t_last, t_start=1)).T  # T makes no sense here but
            e_last[e_spikes] = e_last[e_spikes]*r**(t-te_last[e_spikes]) + ec[t][e_spikes] / float(kd)
            te_last[e_spikes] = t
    return np.cumsum(dws, axis=0)



def pd_weight_grads_past_reloaded(xc, ec, kp, kd):

    r = kd/float(kp+kd)
    n_samples = xc.shape[0]
    assert n_samples == ec.shape[0]
    n_in = xc.shape[1]
    n_out = ec.shape[1]
    dws = np.zeros((n_samples, n_in, n_out))
    tx_last = np.zeros(n_in)
    te_last = np.zeros(n_out)
    x_last = np.zeros(n_in)
    e_last = np.zeros(n_out)
    for t in xrange(n_samples):
        x_spikes = xc[t] != 0









def pd_weight_grads_mult(xc, ec, kp, kd):
    """
    Efficiently compute the weights over time for a system recieving sparse inputs xc and ec.

    We use a slightly different approach this time - instead of keeping the last values and spike
    times for each neuron, we

    :param xc: An (n_samples, n_in) array of input spikes
    :param ec:  An (n_samples, n_in) array of error_spikes
    :param kp: A scalar kp value
    :param kd: A scalar kd value
    :return: An (n_samples, n_in, n_out) array of weight updates
    """
    # TODO: Make this work and pass The Test
    r = kd/float(kp+kd)
    kd=float(kd)
    n_samples = xc.shape[0]
    assert n_samples == ec.shape[0]
    n_in = xc.shape[1]
    n_out = ec.shape[1]
    dws = np.zeros((n_samples, n_in, n_out))
    xr = np.zeros(n_in)
    xi = np.zeros(n_in)
    er = np.zeros(n_out)
    ei = np.zeros(n_out)
    for t in xrange(n_samples):
        xr = r*xr + xc[t]/kd**2
        er = r*er + ec[t]/kd**2
        xi = r*xi*(1-(ec[t]!=0)) + xr
        ei = r*ei*((1-xc[t]!=0)) + er
        # dws[t] = xc[t, :, None]*ei[None, :] + xi[:, None]*ec[t, None, :]
        dws[t] = xc[t, :, None]*ei[None, :] + xi[:, None]*ec[t, None, :]
    return np.cumsum(dws, axis=0)*kd


def pd_weight_grads_future(xc, ec, kp, kd):
    r = kd/float(kp+kd)
    kd=float(kd)
    scale_factor = 1./float(kp**2 + 2*kp*kd)
    n_samples = xc.shape[0]
    assert n_samples == ec.shape[0]
    n_in = xc.shape[1]
    n_out = ec.shape[1]
    dws = np.zeros((n_samples, n_in, n_out))
    xr = np.zeros(n_in)
    er = np.zeros(n_out)
    for t in xrange(n_samples):
        xr *= r
        er = er*r + ec[t]
        dws[t] = (xc[t][:, None]*er[None, :] + xr[:, None]*ec[t][None, :]) * scale_factor
        xr += xc[t]
        # er += ec[t]
    return np.cumsum(dws, axis=0)

