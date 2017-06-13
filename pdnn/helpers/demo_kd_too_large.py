from artemis.experiments.experiment_record import experiment_function, ExperimentFunction
from artemis.experiments.ui import browse_experiments
from artemis.general.mymath import cosine_distance, sqrtspace, point_space
from artemis.general.progress_indicator import ProgressIndicator
import matplotlib.pyplot as plt
import numpy as np
from artemis.plotting.pyplot_plus import non_uniform_imshow, remove_x_axis
from pdnn.helpers.pid_encoder_decoder import lowpass_random, pid_encode, pid_decode
#
# import matplotlib
# print matplotlib.__version__

def display_results(((x, w, xw, xcq, xcn, xwq, xwn), (distance_mat_nonquantized, distance_mat_quantized, distance_mat_recon, n_spikes), (kp, kd, kd_values, kp_values)), include_ops = True):

    plt.figure(figsize=(11,4))
    plt.subplots_adjust(wspace=.5, hspace=0.3, left=0.05, right=.95)

    vmin = min(np.min(distance_mat_quantized), np.min(distance_mat_nonquantized))

    dims = (2, 4) if include_ops else (2, 3)
    ax=plt.subplot2grid(dims, (0, 0), colspan=2)
    plt.plot(x, label='$x_t$')
    plt.plot(xcq, label='$Q(enc(x_t))$', color='k')
    plt.plot(w, label='$w_t$')
    plt.grid(); plt.legend()
    plt.tick_params(axis='x', labelbottom='off')
    plt.title('Example: $k_p={}$, $k_d={}$'.format(kp, kd))

    plt.subplot2grid(dims, (1, 0), colspan=2, sharex=ax)
    plt.plot(xw, label='$x_t\cdot w_t$')
    plt.plot(xwn, label='$dec(enc(x_t)\cdot w_t)$')
    plt.plot(xwq, label='$dec(Q(enc(x_t))\cdot w_t)$')
    plt.grid(); plt.legend()
    plt.xlabel('t')

    # Zoom to a snippet:
    ax.set_xlim(len(x)-500, len(x))

    plt.subplot2grid(dims, (0, 2))
    # plt.imshow(distance_mat_nonquantized[::-1, :], extent=kd_scan_range + kp_scan_range, aspect='auto')
    non_uniform_imshow(distance_mat_nonquantized, x_locs=kd_values, y_locs=kp_values, aspect='auto', vmin=vmin, vmax=1, format_str='{:.2g}')
    plt.ylabel('$k_p$')
    plt.tick_params(axis='x', labelbottom='off')
    # plt.tick_params(axis='y', rotation=90)
    # plt.setp(plt.yticks()[1], rotation=90)
    plt.title('$S_C(x\odot w, dec(enc(x)\odot w))$')
    plt.colorbar()

    plt.subplot2grid(dims, (0, 3))
    # plt.imshow(distance_mat_nonquantized[::-1, :], extent=kd_scan_range + kp_scan_range, aspect='auto')
    non_uniform_imshow(distance_mat_recon, x_locs=kd_values, y_locs=kp_values, aspect='auto', vmin=vmin, vmax=1, format_str='{:.2g}')
    plt.ylabel('$k_p$')
    plt.tick_params(axis='x', labelbottom='off')
    # plt.tick_params(axis='y', rotation=90)
    # plt.setp(plt.yticks()[1], rotation=90)
    plt.tick_params(axis='y', labelleft='off')
    plt.title('$S_C(x\odot w, dec(Q(enc(x)))\odot w)$')
    plt.colorbar()

    plt.subplot2grid(dims, (1, 2))
    # plt.imshow(distance_mat_quantized[::-1, :], extent=kd_scan_range + kp_scan_range, aspect='auto')
    non_uniform_imshow(distance_mat_quantized, x_locs=kd_values, y_locs=kp_values, aspect='auto', vmin=vmin, vmax=1, format_str='{:.2g}')
    plt.xlabel('$k_d$')
    # plt.tick_params(axis='y', rotation=90)
    plt.ylabel('$k_p$')
    plt.title('$S_C(x\odot w, dec(Q(enc(x))\odot w))$')
    plt.colorbar()

    if include_ops:
        plt.subplot2grid(dims, (1, 3))
        non_uniform_imshow(np.log10(n_spikes), x_locs=kd_values, y_locs=kp_values, aspect='auto', format_str='{:.2g}')
        plt.tick_params(axis='y', labelleft='off')
        plt.xlabel('$k_d$')
        plt.title('$log_{10}(N_{ops})$')
        plt.colorbar()

    plt.show()


@ExperimentFunction(display_function=display_results)
def demo_kd_too_large(
        n_steps=20000,
        kp=.01,
        kd=1.,
        kp_scan_range = (.001, .1),
        kd_scan_range = (.1, 10),
        n_k_points = 32,
        x_cutoff = 0.01,
        w_cutoff = 0.002,
        w_fixed = False,
        k_spacing = 'log',
        seed = 1238
        ):
    """
    We have time varying signals x, w.  See how different choices of kp, kd, and quantization affect our
    ability to approximate the time-varying quantity x*w.
    """

    rng = np.random.RandomState(seed)

    x = lowpass_random(n_samples=n_steps, cutoff=x_cutoff, normalize=True, rng=rng)
    w = lowpass_random(n_samples=n_steps, cutoff=w_cutoff, normalize=True, rng=rng) if not w_fixed else np.ones(n_steps)
    x_w = x*w

    distance_mat_nonquantized = np.zeros((n_k_points, n_k_points))
    distance_mat_quantized = np.zeros((n_k_points, n_k_points))
    distance_mat_recon = np.zeros((n_k_points, n_k_points))
    n_spikes = np.zeros((n_k_points, n_k_points))

    pi = ProgressIndicator(n_k_points**2)

    kp_values = point_space(kp_scan_range[0], kp_scan_range[1], n_points=n_k_points, spacing=k_spacing)
    kd_values = point_space(kd_scan_range[0], kd_scan_range[1], n_points=n_k_points, spacing=k_spacing)

    for i, kpi in enumerate(kp_values):
        for j, kdj in enumerate(kd_values):
            pi.print_update(i*n_k_points+j)
            x_enc = pid_encode(x, kp=kpi, kd=kdj, quantization=None)
            x_enc_quantized = pid_encode(x, kp=kpi, kd=kdj, quantization='herd')
            x_enc_w = pid_decode(x_enc*w, kp=kpi, kd=kdj)
            x_enc_quantized_w_dec = pid_decode(x_enc_quantized*w, kp=kpi, kd=kdj)
            x_enc_quantized_dec_w = pid_decode(x_enc_quantized, kp=kpi, kd=kdj)*w
            distance_mat_nonquantized[i, j] = cosine_distance(x_w, x_enc_w)
            distance_mat_quantized[i, j] = cosine_distance(x_w, x_enc_quantized_w_dec)
            distance_mat_recon[i, j] = cosine_distance(x_w, x_enc_quantized_dec_w)
            n_spikes[i,j] = np.abs(x_enc_quantized).sum()

    x_enc_quantized = pid_encode(x, kp=kp, kd=kd, quantization='herd')
    x_enc = pid_encode(x, kp=kp, kd=kd, quantization=None)
    xwq = pid_decode(x_enc_quantized*w, kp=kp, kd=kd)
    xwn = pid_decode(x_enc*w, kp=kp, kd=kd)

    return (x, w, x_w, x_enc_quantized, x_enc, xwq, xwn), (distance_mat_nonquantized, distance_mat_quantized, distance_mat_recon, n_spikes), (kp, kd, kd_values, kp_values)


if __name__ == '__main__':

    # browse_experiments()
    # demo_kd_too_large.add_variant(w_fixed=True).run()
    # demo_kd_too_large.add_variant(w_fixed=True).run()
    demo_kd_too_large.display_last()

