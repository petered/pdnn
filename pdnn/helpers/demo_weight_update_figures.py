import matplotlib
from artemis.plotting.expanding_subplots import vstack_plots, add_subplot
from plato.tools.pretrained_networks.vggnet import im2vgginput

from pdnn.helpers.pid_encoder_decoder import lowpass_random, pid_encode, pid_decode
import numpy as np
from matplotlib import pyplot as plt

# im2vgginput()

def demo_weight_update_figures(
        n_samples = 1000,
        seed=1278,
        kpx=.0015,
        kdx=.5,
        kpe=.0015,
        kde=.5,
        warmup=500,
        future_fill_settings = dict(color='lightsteelblue', hatch='//', edgecolor='w'),
        past_fill_settings = dict(color='lightpink', hatch='\\\\', edgecolor='w'),
        plot=True
        ):
    """
    What this test shows:
    (1) The FutureWeightGradCalculator indeed perfectly calculates the product between the two reconstructions
    (2) The FutureWeightGradCalculator with "true" values plugged in place of reconstructions isn't actually all that great.
    (3) We implemented the "true" value idea correctly, because we try plugging in the reconstructions it is identical to actually using the reconstructions.
    """
    rng = np.random.RandomState(seed)

    linewidth=2

    matplotlib.rcParams['hatch.color'] = 'w'
    matplotlib.rcParams['hatch.linewidth'] = 2.

    rx = kdx/float(kpx+kdx)
    re = kde/float(kpe+kde)

    t = np.arange(n_samples)
    x = lowpass_random(n_samples+warmup, cutoff=0.0003, rng=rng, normalize=True)[warmup:]
    e = lowpass_random(n_samples+warmup, cutoff=0.0003, rng=rng, normalize=True)[warmup:]
    if x.mean()<0:
        x=-x
    if e.mean()<0:
        e=-e
    # x[-int(n_samples/4):]=0
    # e[-int(n_samples/4):]=0
    xc = pid_encode(x, kp=kpx, kd=kdx, quantization='herd')
    ec = pid_encode(e, kp=kpe, kd=kde, quantization='herd')

    xd = pid_decode(xc, kp=kpx, kd=kdx)
    ed = pid_decode(ec, kp=kpe, kd=kde)
    w_true = x*e
    w_recon = xd*ed

    fig = plt.figure(figsize=(7, 3))
    with vstack_plots(grid=False, sharex=False, spacing=0.05, xlabel='t', xlim=(0, n_samples), show_x=False, show_y=False, left=0.01, right=0.98, top=.96, bottom=.08):

        ix = np.nonzero(ec)[0][1]

        future_top_e = ed[ix]*re**(np.arange(n_samples-ix))
        future_bottom_e = ed[ix-1]*re**(np.arange(n_samples-ix)+1)
        future_top_x = xd[ix]*rx**(np.arange(n_samples-ix))

        ix_xlast = np.nonzero(xc[:ix])[0][-1]

        past_top_x = xd[ix_xlast:ix]
        past_top_e = ed[ix_xlast:ix]

        past_top_area = past_top_e*past_top_x

        future_top_area = future_top_e*future_top_x
        future_bottom_area = future_bottom_e*future_top_x

        ax0=ax=add_subplot()
        plt.plot(x, label='$x$', linewidth=linewidth)
        plt.plot(xc, color='k', label='$\\bar x$', linewidth=linewidth+1)
        plt.plot(xd, label='$\hat x$', linewidth=linewidth)
        ax.fill_between(t[ix_xlast:ix], 0., past_top_x, **past_fill_settings)
        ax.fill_between(t[ix:], 0., future_top_x, **future_fill_settings)
        plt.legend(loc='upper left')
        plt.axhline(0, color='k', linewidth=2)

        # ax.arrow(ix-10, 1, ix, 00, head_width=0.05, head_length=0.1, fc='k', ec='k')
        # ax.set_ylim(bottom=-.5, top=4)

        ax1=ax=add_subplot()
        plt.plot(e, label='$e$', linewidth=linewidth)
        plt.plot(ec, color='k', label='$\\bar e$', linewidth=linewidth+1)
        plt.plot(ed, label='$\hat e$', linewidth=linewidth)
        ax.fill_between(t[ix_xlast:ix], 0., past_top_e, **past_fill_settings)
        ax.fill_between(t[ix:], future_bottom_e, future_top_e, **future_fill_settings)
        plt.legend(loc='upper left')
        ax.annotate('spike', xy=(ix+4, 0.4), xytext=(ix+70, 1.), fontsize=10, fontweight='bold',
            arrowprops=dict(facecolor='black', shrink=0.05),
            )


        # plt.ylim(-.5, 4)

        ax2=ax=add_subplot()
        plt.plot(w_true, linewidth=linewidth, label='$\\frac{\partial \mathcal{L}}{\partial w}_t=x_t e_t$')
        plt.plot(w_recon, linewidth=linewidth, label='$\widehat{\\frac{\partial \mathcal{L}}{\partial w}}_t = \hat x_t \hat e_t$')
        ax.fill_between(t[ix_xlast:ix], 0., past_top_area, label='$\widehat{\\frac{\partial \mathcal{L}}{\partial w}}_{t,past}$', **past_fill_settings)
        ax.fill_between(t[ix:], future_bottom_area, future_top_area, label='$\widehat{\\frac{\partial \mathcal{L}}{\partial w}}_{t,future}$', **future_fill_settings)
        plt.axhline(0, color='k', linewidth=2)
        plt.legend(loc='upper left', ncol=2)

    ax0.set_xlim(0, n_samples*3/4)
    ax1.set_xlim(0, n_samples*3/4)
    ax2.set_xlim(0, n_samples*3/4)
    ax0.set_ylim(-.5, 4)
    ax1.set_ylim(-.5, 4)



        # add_subplot()
        # plt.plot(np.cumsum(x*e))
        # plt.plot(w_recon[:, 0, 0])

    plt.show()

if __name__ == '__main__':
    demo_weight_update_figures()