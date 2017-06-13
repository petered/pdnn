from artemis.plotting.expanding_subplots import add_subplot, vstack_plots, hstack_plots
import numpy as np
from matplotlib import pyplot as plt

from pdnn.helpers.efficient_pd_weight_updates import pd_weight_grads
from pdnn.helpers.pid_encoder_decoder import pid_decode



def demo_plot_stdp(kp=0.1, kd=2):

    t=np.arange(-100, 101)
    r = kd/float(kp+kd)

    kbb = r**(np.abs(t))
    k_classic = kbb*np.sign(t)

    plt.figure(figsize=(6, 2))
    with hstack_plots(spacing=0.1, bottom=0.1, left=0.05, right=0.98, xlabel='$t_{post}-t_{pre}$', ylabel='$\Delta w$', sharex=False, sharey=False, show_x=False, remove_ticks=False, grid=True):
        ax=add_subplot()


        plt.plot(t, -kbb)
        plt.title('$sign(\\bar x_t)=sign(\\bar e_t)$')
        plt.xlabel('$t_{post}-t_{pre}$')

        add_subplot()
        plt.plot(t, kbb)
        plt.title('$sign(\\bar x_t)\\neq sign(\\bar e_t)$')

        add_subplot()
        plt.title('Classic STDP Rule')
        plt.plot(t, k_classic)

    ax.tick_params(axis='y', labelleft='off')
    ax.tick_params(axis='x', labelbottom='off')

    plt.show()


def demo_pd_stdp_equivalence(kp=0.03, kd=2., p=0.05, n_samples = 2000):

    r = kd/float(kp+kd)

    rng = np.random.RandomState(1236)
    x_spikes = rng.choice((-1, 0, 1), size=n_samples, p=[p/2, 1-p, p/2])
    e_spikes = rng.choice((-1, 0, 1), size=n_samples, p=[p/2, 1-p, p/2])

    t_k = np.linspace(-5, 5, 201)

    x_spikes[n_samples*3/4:] = 0
    e_spikes[n_samples*3/4:] = 0


    x_spikes[n_samples*7/8]=1


    t = np.arange(-500,  501)
    k = r**t * (t>=0)

    x_hat = np.convolve(x_spikes, k, mode='same')/(kp+kd)
    e_hat = np.convolve(e_spikes, k, mode='same')/(kp+kd)

    x_hat = pid_decode(x_spikes, kp=kp, kd=kd)
    e_hat = pid_decode(e_spikes, kp=kp, kd=kd)


    # kbb = k+(r**(-t) * (t<0))
    kbb = r**(np.abs(t))

    x_conv_kbb = np.convolve(x_spikes, kbb, mode='same')
    sig_future =(x_hat*e_spikes + x_spikes*e_hat - x_spikes*e_spikes/(kp+kd)) * (kp+kd)/float(kp**2 + 2*kp*kd)
    sig_stdp = x_conv_kbb*e_spikes * 1./float(kp**2 + 2*kp*kd)

    sig_past = pd_weight_grads(xc=x_spikes[:, None], ec=e_spikes[:, None], kp=kp, kd=kd)[:, 0, 0]


    plt.figure(figsize=(10, 6))
    with vstack_plots(grid=False, xlabel='t', show_x = False, show_y=False, spacing=0.05, left=0.1, right=0.93, top=0.95):

        add_subplot()
        plt.plot(x_spikes, label='$\\bar x_t$')
        plt.plot(x_hat, label='$\hat x_t$')
        plt.axhline(0, linewidth=2, color='k')
        plt.legend(loc = 'lower right')
        plt.ylim(-2, 2)
        plt.ylabel('Presynaptic\nSignal')
        # plt.ylabel('$\\bar x_t$')

        add_subplot()
        plt.plot(e_spikes, label='$\\bar e_t$')
        plt.plot(e_hat, label='$\hat e_t$')
        plt.axhline(0, linewidth=2, color='k')
        plt.legend(loc = 'lower right')
        plt.ylim(-2, 2)
        plt.ylabel('Postsynaptic\nSignal')
        # plt.ylabel('$\\bar e_t$')

        # add_subplot()
        # plt.plot(x_hat, label='xk', marker='.')
        # plt.plot(e_hat-2*np.max(np.abs(x_hat)), label='yk', marker='.')
        # add_subplot()
        # plt.plot(x_conv_kbb, marker='.', label='x * kbb')
        # add_subplot()
        # plt.plot(x_conv_kbb*e_spikes, label='$(x * kbb) \odot y$')
        # plt.plot(sig_future, label = 'xk*y + x*yk')
        # sig_kbb = -x_conv_kbb*ym + x_conv_kbb*yp
        # plt.plot(sig_kbb, label='new', linestyle='--')
        # sig_stdp= -x_conv_kstdp*ym + x_conv_kstdp*yp
        # sig_stdp = x_conv_kstdp * y
        # plt.plot(sig_stdp, label='new', linestyle='--')
        # plt.legend()

        add_subplot()
        plt.plot(-np.cumsum(x_hat*e_hat), label='recon')
        plt.plot(-np.cumsum(sig_stdp), label='STDP')
        plt.plot(-sig_past, label='past')
        plt.plot(-np.cumsum(sig_future), label='future')
        plt.axhline(0, linewidth=2, color='k')
        plt.ylabel('$\sum_{\\tau=0}^t \Delta w_\\tau$')

        # plt.plot(np.cumsum(sig_stdp))

        plt.legend(loc = 'lower right')


    # add_subplot(layout='v')
    # plt.plot(xkky, label='xkky', linestyle='--')
    # plt.legend()

    # plt.plot(kk)
    # plt.plot(k)

    plt.show()

if __name__ == '__main__':
    demo_pd_stdp_equivalence()
    # demo_plot_stdp()