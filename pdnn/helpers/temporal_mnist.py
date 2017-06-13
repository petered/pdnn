import numpy as np

from artemis.fileman.disk_memoize import memoize_to_disk_and_cache
from artemis.general.should_be_builtins import bad_value
from artemis.ml.datasets.datasets import DataSet
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.general.progress_indicator import ProgressIndicator


__author__ = 'peter'


def temporalize(x, smoothing_steps, distance='L1'):
    """
    :param x: An (n_samples, n_dims) dataset
    :return: A (n_samples, ) array of indexes that can be used to shuffle the input for temporal smoothness.
    """
    x_flat = x.reshape(x.shape[0], -1)
    index_buffer = np.arange(1, smoothing_steps+1)
    next_sample_buffer = x_flat[1:smoothing_steps+1].copy()
    # Technically, we could do this without a next_sample_buffer (and only an index_buffer), but it would require
    # repeatedly accessing a bunch of really scattered memory, so we do it this way.
    shuffling_indices = np.zeros(len(x), dtype=int)
    rectifier = np.abs if distance=='L1' else np.square if distance=='L2' else bad_value(distance)
    p=ProgressIndicator(len(x), name = 'Temporalize')
    current_index = 0
    for i in xrange(len(x)):
        shuffling_indices[i] = current_index
        closest = np.argmin(rectifier(x_flat[current_index]-next_sample_buffer).sum(axis=1))
        current_index = index_buffer[closest]
        weve_aint_done_yet = i+smoothing_steps+1 < len(x)
        next_index = i+smoothing_steps+1
        next_sample_buffer[closest] = x_flat[next_index] if weve_aint_done_yet else float('inf')
        index_buffer[closest] = next_index if weve_aint_done_yet else -1
        p()
    return shuffling_indices


@memoize_to_disk_and_cache
def get_temporal_mnist_dataset(smoothing_steps=1000, **mnist_kwargs):

    tr_x, tr_y, ts_x, ts_y = get_mnist_dataset(**mnist_kwargs).xyxy
    tr_ixs = temporalize(tr_x, smoothing_steps=smoothing_steps)
    ts_ixs = temporalize(ts_x, smoothing_steps=smoothing_steps)
    return DataSet.from_xyxy(tr_x[tr_ixs], tr_y[tr_ixs], ts_x[ts_ixs], ts_y[ts_ixs])
