from artemis.general.mymath import cosine_distance

from pdnn.helpers.pid_encoder_decoder import lowpass_random, pid_encode, pid_decode, Herder
from matplotlib import pyplot as plt
import numpy as np


def demo_visualize_k_effects(
        kps = [0., 0.01, .1, 2.],
        kds = [0, 1., 4.],
        cutoff=0.005,
        n_samples=550,
        s_as_triangles = False,
        seed=1234
        ):

    x = lowpass_random(n_samples = n_samples, cutoff=cutoff, rng=seed, normalize=True)

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.08, right=.98, top=.92)
    ax=plt.subplot2grid((len(kps), len(kds)), (0, 0))
    for i, kp in enumerate(kps):
        for j, kd in enumerate(kds):
            xe = pid_encode(x, kp=kp, kd=kd)
            h = Herder()
            xc = [h(xet) for xet in xe]
            xd = pid_decode(xc, kp=kp, kd=kd)
            this_ax = plt.subplot2grid((len(kps), len(kds)), (len(kps)-i-1, j), sharex=ax, sharey=ax)
            plt.plot(xd, color='C1', label='$\hat x_t$')
            # plt.text(0, -0.1, '$Sc(x,\hat x)={:.2g},|s|={}$'.format(cosine_distance(x, xd), np.sum(np.abs(xc))))

            # plt.text(0.01, .99, '$Sc(x,\hat x)={:.3g},|s|={}$'.format(cosine_distance(x, xd), int(np.sum(np.abs(xc)))),
            #          ha='left', va='top', transform=this_ax.transAxes, bbox=dict(boxstyle='square', facecolor='w', alpha=0.7, pad=0))
            # plt.text(0.01, .99, '$|x-\hat x|^2={:.2g},|s|={}$'.format(np.sqrt(((x-xd)**2).mean()), int(np.sum(np.abs(xc)))),
            #          ha='left', va='top', transform=this_ax.transAxes, bbox=dict(boxstyle='square', facecolor='w', alpha=0.8, pad=0))
            # plt.text(0.01, .01, '$\left<|x_t-\hat x_t|\\right>_t={:.2g},  \Sigma_t|s_t|={}$'.format(np.abs(x-xd).mean(), int(np.sum(np.abs(xc)))),
            #          ha='left', va='bottom', transform=this_ax.transAxes, bbox=dict(boxstyle='square', facecolor='w', edgecolor='none', alpha=0.8, pad=0.0))
            plt.text(.01, .01, '$\left<|x_t-\hat x_t|\\right>_t={:.2g}, \;\;\;  N={}$'.format(np.abs(x-xd).mean(), int(np.sum(np.abs(xc)))),
                     ha='left', va='bottom', transform=this_ax.transAxes, bbox=dict(boxstyle='square', facecolor='w', edgecolor='none', alpha=0.8, pad=0.0))
            # plt.text(0.5, 0.5,'matplotlib',
            #      horizontalalignment='center',
            #      verticalalignment='center',
            #      transform = ax.transAxes)

            # plt.plot(xe, color='C4', label='$a_t$')
            if s_as_triangles:
                up_spikes = np.nonzero(xc>0)[0]
                down_spikes = np.nonzero(xc<0)[0]
                plt.plot(up_spikes, np.zeros(up_spikes.shape), '^', color='k', label='$s_t^+$')
                plt.plot(down_spikes, np.zeros(down_spikes.shape), 'v', color='r', label='$s_t^-$')
            else:
                plt.plot(xc, color='k', label='$s_t$')
            plt.plot(x, color='C0', label='$x_t$')
            plt.grid()
            if i>0:
                plt.tick_params('x', labelbottom='off')
            else:
                plt.xlabel('$k_d={}$'.format(kd))
            if j>0:
                plt.tick_params('y', labelleft='off')
            else:
                plt.ylabel('$k_p={}$'.format(kp))

    ax.set_xlim(0, n_samples)
    ax.set_ylim(np.min(x)*1.1, np.max(x)*1.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, ncol=len(handles[::-1]))
    plt.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, ncol=len(handles[::-1]), loc='upper right')
    plt.show()


if __name__ == '__main__':
    demo_visualize_k_effects()


