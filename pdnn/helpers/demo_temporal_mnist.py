import numpy as np

from artemis.plotting.db_plotting import dbplot, hold_dbplots
from pdnn.helpers.temporal_mnist import get_temporal_mnist_dataset
from artemis.ml.datasets.mnist import get_mnist_dataset

__author__ = 'peter'


def demo_temporal_mnist(n_samples = None, smoothing_steps = 200):
    _, _, original_data, original_labels = get_mnist_dataset(n_training_samples=n_samples, n_test_samples=n_samples).xyxy
    _, _, temporal_data, temporal_labels = get_temporal_mnist_dataset(n_training_samples=n_samples, n_test_samples=n_samples, smoothing_steps=smoothing_steps).xyxy
    for ox, oy, tx, ty in zip(original_data, original_labels, temporal_data, temporal_labels):
        with hold_dbplots():
            dbplot(ox, 'sample', title = str(oy))
            dbplot(tx, 'smooth', title = str(ty))


def demo_plot_temporal_mnist(n_rows=8, n_cols=16, smoothing_steps=1000):
    _, _, temporal_data, temporal_labels = get_temporal_mnist_dataset(smoothing_steps=smoothing_steps).xyxy
    stride = len(temporal_data)/n_rows
    starts = np.arange(0, stride*n_rows, stride)
    data = np.array([temporal_data[s:s+n_cols] for s in starts]).swapaxes(0, 1)
    dbplot(data, 'Temporal MNIST', plot_type = 'pic', hang=True)



if __name__ == '__main__':
    # demo_plot_temporal_mnist()
    demo_temporal_mnist(n_samples=None, smoothing_steps = 1000)
