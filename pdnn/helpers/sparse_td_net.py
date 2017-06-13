from artemis.general.should_be_builtins import izip_equal
from pdnn.helpers.forward_pass import scaled_quantized_forward_pass, sparse_nn_flop_count
from artemis.ml.tools.neuralnets import activation_function
from pdnn.helpers.quantization import quantize_sequence

__author__ = 'peter'
import numpy as np


def sparse_temporal_forward_pass(inputs, weights, biases = None, scales = None, hidden_activations='relu', output_activations = 'relu', quantization_method = 'herd', rng=None):
    """
    Feed a sequence of inputs into a sparse temporal difference net and get the resulting activations.

    :param inputs: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param scales: An optional (len(weights)) list of (w.shape[0]) scales to scale each layer before rounding.
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activations: Indicates the output layer activation function
    :return: activations:
        A len(weights)*3+1 list of (n_frames, n_dims) activations.
        Elements [::3] will be a length(w)+1 list containing the input to each rounding unit, and the final output
        Elements [1::3] will be the length(w) rounded "spike" signal.
        Elements [2::3] will be the length(w) inputs to each nonlinearity
    """
    activations = [inputs]
    if biases is None:
        biases = [0]*len(weights)
    if scales is None:
        scales = [1.]*len(weights)
    else:
        assert len(scales) in (len(weights), len(weights)+1)
    real_activations = inputs
    for w, b, k in zip(weights, biases, scales):
        deltas = np.diff(np.insert(real_activations, 0, 0, axis=0), axis=0)  # (n_steps, n_in)
        spikes = quantize_sequence(k*deltas, method=quantization_method, rng=rng)  # (n_steps, n_in)
        delta_inputs = (spikes/k).dot(w)  # (n_steps, n_out)
        cumulated_inputs = np.cumsum(delta_inputs, axis=0)+b  # (n_steps, n_out)
        real_activations = activation_function(cumulated_inputs, output_activations if w is weights[-1] else hidden_activations)  # (n_steps, n_out)
        activations += [spikes, cumulated_inputs, real_activations]
    if len(scales)==len(weights)+1:
        activations[-1]*=scales[-1]
    return activations


def tdnet_forward_pass_cost_and_output(inputs, weights, scales=None, version = 'td', computation_calc='additions', **other_kwargs):
    """
    Do a forward pass of a discretized Temporal Difference network, and return the (pseudo) computational cost and final output.
    (This is a clone of quantized_forward_pass_cost_and_output, but for the temporal difference net)

    :param inputs:
    :param weights:
    :param scales:
    :param computation_calc: 'adds' or 'multiplyadds'
    :param other_kwargs:
    :return:
    """
    assert version in ('td', 'round')
    if version == 'td':
        activations = sparse_temporal_forward_pass(inputs = inputs, weights=weights, scales=scales, **other_kwargs)
        # dbplot(activations[4][:100], 'td-acts')
    else:
        activations = scaled_quantized_forward_pass(inputs = inputs, weights=weights, scales=scales, **other_kwargs)
        # dbplot(activations[4][:100], 'round-acts')
    spike_activations = activations[1::3]
    # assert all(np.array_equal(np.round(a), a) for a in spike_activations), "So-called 'spikes' are not integer"

    if isinstance(computation_calc, (list, tuple)):
        n_ops = [sparse_nn_flop_count(spike_activations, [w.shape[1] for w in weights], mode=m) for m in computation_calc]
    else:
        n_ops = sparse_nn_flop_count(spike_activations, [w.shape[1] for w in weights], mode=computation_calc)
    # n_ops = sum(np.abs(s).sum()*w.shape[1] for s, w in izip_equal(spike_activations, weights))
    return n_ops, activations[-1]
