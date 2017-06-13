import numpy as np

from artemis.general.numpy_helpers import get_rng

from artemis.general.should_be_builtins import izip_equal, bad_value
from artemis.ml.tools.neuralnets import activation_function, forward_pass_activations
from pdnn.helpers.quantization import sequential_quantize, quantize


__author__ = 'peter'


def scaled_herding_forward_pass(weights, scales, input_data, n_steps):
    """
    Do a forward pass with scaled units.
    :param weights: A length:L list of weight matrices
    :param scales: A length:L+1 list of scale vectors
    :param input_data: An (n_samples, n_dims) array of input data
    :param n_steps: Number of steps to run for
    :param rng: A random number generator or see    d
    :return: A (n_samples, n_steps, n_outputs) integar array representing the output spike count in each time bin
    """
    assert all(w_in.shape[1]==w_out.shape[0] for w_in, w_out in zip(weights[:-1], weights[1:]))
    assert len(scales) == len(weights)+1
    assert all(s.ndim==1 for s in scales)
    assert all(w.ndim==2 for w in weights)
    assert all(len(s_in)==w.shape[0] and len(s_out)==w.shape[1] for w, s_in, s_out in zip(weights, scales[:-1], scales[1:]))
    assert input_data.shape[1] == len(scales[0])

    # scaled_weights = [s_in[:, None]**-1 * w * s_out[None, :] for s_in, w, s_out in zip(scales[:-1], weights, scales[1:])]
    spikes = sequential_quantize(input_data*scales[0], n_steps=n_steps)
    for s, w in zip(scales[1:], weights):
        spikes = sequential_quantize(spikes.dot(w)*s)
        # spikes = sequential_quantize(spikes.dot(w/s_pre[:, None])*s_post)
    return spikes/scales[-1]


def simple_rounded_relu_forward_pass_guts(weights, input_data, n_steps):
    activation = np.round(n_steps*input_data)/float(n_steps)
    activations = [activation]
    for w in weights:
        u = np.maximum(0, activation.dot(w))
        activation = np.round(u*n_steps)/float(n_steps)
        activations.append(activation)
    return activations


def simple_relu_forward_pass(weights, input_data):
    activation = input_data
    for w in weights:
        activation = np.maximum(0, activation.dot(w))
    return activation


def stochastically_rounded_relu_forward_pass_guts(weights, input_data, n_steps, rng = None):
    rng = get_rng(rng)
    activation = np.round(n_steps*input_data + rng.uniform(-.5, .5, size=input_data.shape))/float(n_steps)
    activations = [activation]
    for w in weights:
        u = np.maximum(0, activation.dot(w))
        activation = np.round(u*n_steps + rng.uniform(-.5, .5, size=u.shape))/float(n_steps)
        activations.append(activation)
    return activations


def forward_pass_and_cost(input_data, weights, biases = None, **forward_pass_kwargs):
    """
    Do a forward pass and return the computaitonal cost.

    :param input_data: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activations: Indicates the output layer activation function
    :return: output, n_dense_flops, n_sparse_flops ... where:
        output is a (input_data.shape[0], weights[-1].shape[1]) array of outputs
        n_dense_flops is the total number of floating point operations (adds and multiples) required in the forward pass
        n_sparse_flops is the total number of operations not-counting zero-activations.
    """
    activations = forward_pass_activations(input_data=input_data, weights=weights, biases=biases, **forward_pass_kwargs)

    # dbplot(activations[2][:100], 'full-acts')

    # full_flops_per_datapoint = [(w.shape[0]*w.shape[1] + (w.shape[0]-1)*w.shape[1]) for w in weights]
    n_dense_flops = sum(w.shape[0]*w.shape[1]+(w.shape[0]-1)*w.shape[1] for w in weights)*input_data.shape[0]
    layerwise_total_nonzero_acts = [(act!=0).sum() for act in activations[:-1:2]]
    n_sparse_flops = sum((actsum*w.shape[1] + (actsum-1)*w.shape[1]) for actsum, w in izip_equal(layerwise_total_nonzero_acts, weights))
    # n_sparse_flops =sum((act!=0).mean(axis=1).sum(axis=0)*full_layer_flops for act, full_layer_flops in izip_equal(activations[:-1:2], full_flops_per_datapoint))
    return activations[-1], n_dense_flops, n_sparse_flops


def quantized_forward_pass(input_data, weights, biases = None, hidden_activations='relu', output_activations = 'relu',
            quantization_method = 'herd', rng=None):
    """
    Return the activations from a forward pass of a ReLU net.
    :param input_data: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activations: Indicates the output layer activation function
    :param quantization_method: The method for quantizing (see function: sequential_quantize)
    :param rng: A random number generator or seed
    :return: activations:
        A len(weights)*3+1 list of (n_frames, n_dims) activations.
        Elements [::3] will be a length(w)+1 list containing the input to each rounding unit, and the final output
        Elements [1::3] will be the length(w) rounded "spike" signal.
        Elements [2::3] will be the length(w) inputs to each nonlinearity
    """
    activations = [input_data]
    if biases is None:
        biases = [0]*len(weights)
    else:
        assert len(biases)==len(weights)
    x = input_data  # (n_samples, n_units)
    for i, (w, b) in enumerate(zip(weights, biases)):
        s = quantize(x, method=quantization_method, rng=rng)
        u = s.dot(w)+b
        x = activation_function(u, output_activations if i==len(weights)-1 else hidden_activations)
        activations += [s, u, x]
    return activations


def scaled_quantized_forward_pass(inputs, weights, scales = None, biases = None, hidden_activations='relu', output_activations = 'relu',
            quantization_method = 'herd', rng=None):
    """
    Return the activations from a forward pass of a ReLU net.
    :param inputs: A (n_frames, n_dims_in) array
    :param weights: A list of (n_dim_in, n_dim_out) weight matrices
    :param biases: An optional (len(weights)) list of (w.shape[1]) biases for each weight matrix
    :param hidden_activations: Indicates the hidden layer activation function
    :param output_activations: Indicates the output layer activation function
    :param quantization_method: The method for quantizing (see function: sequential_quantize)
    :param rng: A random number generator or seed
    :return: activations:
        A len(weights)*3+1 list of (n_frames, n_dims) activations.
        Elements [::3] will be a length(w)+1 list containing the input to each rounding unit, and the final output
        Elements [1::3] will be the length(w) rounded "spike" signal.
        Elements [2::3] will be the length(w) inputs to each nonlinearity
    """
    rng = get_rng(rng)
    activations = [inputs]
    if biases is None:
        biases = [0]*len(weights)
    else:
        assert len(biases)==len(weights)

    if scales is None:
        scales = [1.]*len(weights)

    x = inputs  # (n_samples, n_units)
    for i, (w, b, k) in enumerate(izip_equal(weights, biases, scales)):
        s = quantize(x*k, method=quantization_method, rng=rng)
        u = (s/k).dot(w)+b
        x = activation_function(u, output_activations if i==len(weights)-1 else hidden_activations)
        activations += [s, u, x]
    return activations


def sparse_nn_flop_count(activations, fanouts, mode = 'adds'):

    assert len(activations)==len(fanouts)
    if mode=='adds':
        assert all(np.array_equal(np.round(a), a) for a in activations)
        n_ops = sum(np.abs(s).sum()*fanout for s, fanout in izip_equal(activations, fanouts))
    elif mode=='multiplyadds':
        n_ops = 2 * sum(np.sum(s != 0) * fanout for s, fanout in izip_equal(activations, fanouts))
    else:
        bad_value(mode)
    return n_ops


def quantized_forward_pass_cost_and_output(inputs, weights, scales, biases=None, quantization_method='round',
        hidden_activations='relu', output_activation = 'relu', computation_calc='adds', seed=None):
    """
    Do a forward pass of a discretized network, and return the (pseudo) computational cost and final output.

    :param inputs: A (n_samples, n_dims) array of inputs
    :param weights: A list of (n_dim_in, n_dim_out) arrays of weights
    :param scales: A list of (w[0].shape[0], w[1].shape[0], ...) scales to multiply/divide by before/after the quantization
    :param quantization_method: The method of quantization/discretization: 'round', 'uniform', None, ....
    :param seed: A random seed or number generator
    :return: n_ops, output_activation: Where:
        n_ops is the (scalar) number of commputations required in the forward pass (only striclty true if scale is 'round', .. otherwise it's some kind of surrogate.
        output_activation: A (n_samples, n_dims) array representing the output activations.
    """
    activations = scaled_quantized_forward_pass(inputs= inputs, weights=weights, biases=biases, scales=scales,
        hidden_activations=hidden_activations, output_activations=output_activation, quantization_method=quantization_method, rng=seed)
    spike_activations = activations[1::3]
    n_ops = sparse_nn_flop_count(spike_activations, [w.shape[1] for w in weights], mode=computation_calc) if quantization_method is not None else None
    return n_ops, activations[-1]
