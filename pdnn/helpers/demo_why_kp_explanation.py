
import numpy as np
from artemis.plotting.expanding_subplots import vstack_plots, add_subplot
from matplotlib import pyplot as plt
from pdnn.helpers.pid_encoder_decoder import lowpass_random, pid_decode, pid_encode


def demo_why_kp_explanation(
        n_steps=2000,
        kd=1.,
        kp_values = [0, 0.01, 0.1],
        x_cutoff = 0.03,
        w_cutoff = 0.01,
        w_fixed = False,
        seed = 1234,
        ):
    """
    We have time varying signals x, w.  See how different choices of kp, kd, and quantization affect our
    ability to approximate the time-varying quantity x*w.
    """

    rng = np.random.RandomState(seed)

    x = lowpass_random(n_samples=n_steps, cutoff=x_cutoff, normalize=True, rng=rng)
    w = lowpass_random(n_samples=n_steps, cutoff=w_cutoff, normalize=True, rng=rng)+1 if not w_fixed else np.ones(n_steps)
    xw = x*w

    plt.figure(figsize=(10, 4))
    with vstack_plots(sharex=True, sharey=True, left=0.09, right=.98, spacing=0.02, remove_ticks=False):
        ax=add_subplot()
        plt.plot(x, label='$x_t$')
        plt.plot(w, label='$w_t$')
        plt.title('In all plots, $k_d={}$'.format(kd), loc='left')
        plt.grid()

        for kp in kp_values:
            s = pid_encode(x, kp=kp, kd=kd, quantization='herd')
            zprime = pid_decode(s*w, kp=kp, kd=kd)
            ax_mult=add_subplot()
            plt.plot(xw, label = '$z_t=x_t\cdot w_t$', color='C2')
            plt.plot(zprime, label='$\hat z_t = dec_{{k_p k_d}}(Q(enc_{{k_p k_d}}(x_t))\cdot w_t)$'.format(kp), color='C3')
            plt.ylabel('$k_p={}$'.format(kp))
            # plt.tick_params(axis='y', labelleft='off')
            # plt.ylim(-4.5, 4.5)
            plt.grid()
        # plt.plot(xw, label = '$z_t$', color='k', linewidth=2)
        plt.xlabel('t')
# plt.legend()


    # ax.set_ylim(-2.7, 2.7)
    ax_mult.set_ylim(-4.5, 4.5)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax_mult.get_legend_handles_labels()
    # plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, ncol=len(handles[::-1]))
    plt.legend(handles+handles2, labels+labels2,bbox_to_anchor=(.99, .99), bbox_transform=plt.gcf().transFigure, ncol=len(handles+handles2), loc='upper right')


    plt.show()


if __name__ == '__main__':
    demo_why_kp_explanation()
