import numpy as np
from artemis.experiments.experiment_record import ExperimentFunction, load_experiment, capture_created_experiments
from artemis.experiments.ui import browse_experiments
from artemis.general.display import IndentPrint
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.train_and_test import train_and_test_online_predictor
from artemis.ml.tools.neuralnets import initialize_network_params
from artemis.plotting.db_plotting import dbplot
from artemis.plotting.expanding_subplots import add_subplot, hstack_plots
from artemis.plotting.pyplot_plus import get_line_color
from matplotlib import pyplot as plt
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import GradientDescent
from si_prefix import si_format
from pdnn.helpers.pdnnet import PDHerdingNetwork, PDEncoderDecoder, PDAdaptiveEncoderDecoder
from pdnn.helpers.temporal_mnist import get_temporal_mnist_dataset
from pdnn.helpers.measure_mnist_results import \
    estimate_energy_cost


def compare_results(results, names = None):

    e_ops = {}
    e_args = {}
    e_energies = {}
    training_scores = {}
    test_scores = {}
    e_epochs = {}
    for exp_name, (learning_curves, op_count_info) in results.iteritems():
        infos, ops = zip(*op_count_info)
        e_epochs[exp_name] = [info.epoch for info in infos]
        ops = np.array(ops)
        ops[:, 1, :] = 0  # Because we don't actually have to do the backward pass of the first layer.
        ops = ops.sum(axis=2).sum(axis=1)
        e_ops[exp_name] = ops
        arg = load_experiment(exp_name).get_args()
        e_args[exp_name] = arg
        if arg['swap_mlp']:
            e_energies[exp_name] = estimate_energy_cost(n_ops = ops, op='mult-add', dtype='int')
            training_scores[exp_name] = learning_curves.get_values(subset='train', prediction_function = None, score_measure=None)
            test_scores[exp_name] = learning_curves.get_values(subset='test', prediction_function = None, score_measure=None)
        else:
            e_energies[exp_name] = estimate_energy_cost(n_ops = ops, op='add', dtype='int')
            training_scores[exp_name] = learning_curves.get_values(subset='train', prediction_function = 'herded', score_measure=None)
            test_scores[exp_name] = learning_curves.get_values(subset='test', prediction_function = 'herded', score_measure=None)

    with hstack_plots(ylabel='Temporal MNIST Score', grid=True, ylim=(85, 102)):
        ax=add_subplot()
        for exp_name in results:
            plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(e_epochs[exp_name], training_scores[exp_name], color=colour, label=names[exp_name]+':Training')
            plt.plot(e_epochs[exp_name], test_scores[exp_name], color=colour, label=names[exp_name]+':Test')
            plt.xlabel('Epoch')
            plt.legend()
            plt.xlim(0, 50)

        ax=add_subplot()
        for exp_name in results:
            plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(e_ops[exp_name]/1e9, training_scores[exp_name], color=colour, label=names[exp_name]+':Training')
            plt.plot(e_ops[exp_name]/1e9, test_scores[exp_name], color=colour, label=names[exp_name]+':Test')
            plt.xlabel('GOps')
            plt.legend()

        ax=add_subplot()
        for exp_name in results:
            plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(e_energies[exp_name]/1e9, training_scores[exp_name], color=colour, label=names[exp_name]+':Training')
            plt.plot(e_energies[exp_name]/1e9, test_scores[exp_name], color=colour, label=names[exp_name]+':Test')
            plt.xlabel('Energies')
            plt.legend()

    plt.show()
    pass


def do_the_figure(mnist_results, temporal_mnist_results, names = None, predict_on='herded', remove_prefix = True, title=None):

    # args = {exp_name: load_experiment(exp_name).get_args() for exp_name in results}

    plots_shape=(2, 3)
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(wspace=0, hspace=0, top=0.92, right=0.95)

    for row, dataset in enumerate(['MNIST', 'Temporal MNIST']):

        results = mnist_results if dataset=='MNIST' else temporal_mnist_results

        e_ops = {}
        e_args = {}
        e_energies = {}
        training_scores = {}
        test_scores = {}
        e_epochs = {}
        for exp_name, (learning_curves, op_count_info) in results.iteritems():
            infos, ops = zip(*op_count_info)
            e_epochs[exp_name] = [info.epoch for info in infos]
            ops = np.array(ops)
            ops[:, 1, :] = 0  # Because we don't actually have to do the backward pass of the first layer.
            ops = ops.sum(axis=2).sum(axis=1)
            e_ops[exp_name] = ops
            arg = load_experiment(exp_name).get_args()
            e_args[exp_name] = arg
            if arg['swap_mlp']:
                e_energies[exp_name] = estimate_energy_cost(n_ops = ops, op='mult-add', dtype='int')
                training_scores[exp_name] = learning_curves.get_values(subset='train', prediction_function = None, score_measure=None)
                test_scores[exp_name] = learning_curves.get_values(subset='test', prediction_function = None, score_measure=None)
            else:
                e_energies[exp_name] = estimate_energy_cost(n_ops = ops, op='add', dtype='int')
                training_scores[exp_name] = learning_curves.get_values(subset='train', prediction_function = predict_on, score_measure=None)
                test_scores[exp_name] = learning_curves.get_values(subset='test', prediction_function = predict_on, score_measure=None)
            print 'Scores for {}\n  Training: {}\n  Test: {}\n  GOps: {}'.format(exp_name, np.max(training_scores[exp_name]), np.max(test_scores[exp_name]), ops.sum()/1e9)

        ylim_args = (91, 101)
        ax=plt.subplot2grid(plots_shape, (row, 0))
        for i, exp_name in enumerate(results):
            # plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            # colour='b'
            plt.semilogx(e_epochs[exp_name], training_scores[exp_name], color=get_line_color(i), label=names[exp_name]+':Training', linestyle='--')
            plt.semilogx(e_epochs[exp_name], test_scores[exp_name], color=get_line_color(i, 'dark'), label=names[exp_name]+':Test')


            ax.grid(True)
            # plt.legend()
            plt.xlim(0, 50)
            if row==0:
                plt.tick_params('x', labelbottom='off')
            else:
                plt.xlabel('Epoch')
            plt.ylabel('% Score on \n{}'.format(dataset))
            plt.ylim(*ylim_args)
            plt.xlim(0.5, 50)

        plt.subplot2grid(plots_shape, (row, 1))
        for i, exp_name in enumerate(results):
            # plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            # colour = 'b'
            plt.semilogx(e_ops[exp_name]/1e9, training_scores[exp_name], color=get_line_color(i), label=names[exp_name]+':Training', linestyle='--')
            plt.semilogx(e_ops[exp_name]/1e9, test_scores[exp_name], color=get_line_color(i, 'dark'), label=names[exp_name]+':Test')
            # plt.legend()
            if row==0:
                plt.tick_params('x', labelbottom='off')
            else:
                plt.xlabel('GOps')
            plt.grid(True)
            plt.tick_params('y', labelleft='off')
            plt.ylim(*ylim_args)
            plt.xlim(5, 1000)

        plt.subplot2grid(plots_shape, (row, 2))
        for i, exp_name in enumerate(results):
            # plt.plot()
            colour = next(plt.gca()._get_lines.prop_cycler)['color']
            # colour = 'b'
            plt.semilogx(e_energies[exp_name]/1e9, training_scores[exp_name], color=get_line_color(i), label=names[exp_name]+':Training', linestyle='--')
            plt.semilogx(e_energies[exp_name]/1e9, test_scores[exp_name], color=get_line_color(i, 'dark'), label=names[exp_name]+':Test')
            # plt.legend()
            if row==0:
                plt.tick_params('x', labelbottom='off')
            else:
                plt.xlabel('Energies (mJ)')
            plt.grid(True)
            plt.tick_params('y', labelleft='off')
            plt.ylim(*ylim_args)
            plt.xlim(.2, 2000)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels,bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, ncol=len(handles[::-1]), loc='upper right')
    plt.show()
    pass


def _ops_per_epoch(info_op_pairs):
    infos, ops = zip(*info_op_pairs)
    return 'GOps/ep: {}'.format(np.sum(ops[-1])/infos[-1].epoch/1e9)


@ExperimentFunction(one_liner_results=lambda (isp, iops): isp.get_summary()+','+_ops_per_epoch(iops), comparison_function=compare_results, is_root=True)
def demo_herding_network(
        kp=.1,
        kd = 1.,
        kp_back=None,
        kd_back=None,
        hidden_sizes = [200, ],
        n_epochs = 50,
        onehot = False,
        parallel=False,
        learning_rate=0.01,
        dataset = 'mnist',
        hidden_activation='relu',
        adaptive = True,
        adaptation_rate = 0.001,
        output_activation = 'softmax',
        loss='nll',
        fwd_quantizer ='herd',
        back_quantizer='same',
        minibatch_size=1,
        swap_mlp = False,
        plot=False,
        test_period=.5,
        grad_calc = 'true',
        rng = 1234
        ):

    dataset = get_mnist_dataset(flat=True, join_train_and_val=True) if dataset == 'mnist' else get_temporal_mnist_dataset(flat=True, join_train_and_val=True)
    if onehot:
        dataset = dataset.to_onehot()
    ws = initialize_network_params(layer_sizes=[28*28]+hidden_sizes+[10], mag='xavier-both', include_biases=False, rng=rng)

    if is_test_mode():
        dataset = dataset.shorten(500)
        n_epochs=0.1
        test_period=0.03

    if kp_back is None:
        kp_back = kp
    if kd_back is None:
        kd_back = kd
    if back_quantizer=='same':
        back_quantizer=fwd_quantizer

    if adaptive:
        encdec = lambda: PDAdaptiveEncoderDecoder(kp=kp, kd=kd, adaptation_rate=adaptation_rate, quantization=fwd_quantizer)
        encdec_back = lambda: PDAdaptiveEncoderDecoder(kp=kp_back, kd=kd_back, adaptation_rate=adaptation_rate, quantization=back_quantizer)
    else:
        encdec = PDEncoderDecoder(kp=kp, kd=kd, quantization=fwd_quantizer)
        encdec_back = PDEncoderDecoder(kp=kp_back, kd=kd_back, quantization=back_quantizer)

    if swap_mlp:
        if not parallel:
            assert minibatch_size==1, "Unfair comparison otherwise, sorry buddy, can't let you do that."
        net = GradientBasedPredictor(
            function = MultiLayerPerceptron.from_weights(
                weights=ws,
                hidden_activations=hidden_activation,
                output_activation=output_activation,
                ),
            cost_function=loss,
            optimizer=GradientDescent(learning_rate),
            )
        prediction_funcs = net.predict.compile()
    else:
        net = PDHerdingNetwork(
            ws = ws,
            encdec = encdec,
            encdec_back = encdec_back,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer=GradientDescent(learning_rate),
            minibatch_size=minibatch_size if parallel else 1,
            grad_calc = grad_calc,
            loss = loss
            )
        noise_free_forward_pass = MultiLayerPerceptron.from_weights(
            weights=[layer.w for layer in net.layers],
            biases=[layer.b for layer in net.layers],
            hidden_activations=hidden_activation,
            output_activation=output_activation
            ).compile()
        prediction_funcs = [('noise_free', noise_free_forward_pass), ('herded', net.predict.compile())]

    op_count_info = []
    def test_callback(info, score):
        if plot:
            dbplot(net.layers[0].w.get_value().T.reshape(-1, 28, 28), 'w0', cornertext='Epoch {}'.format(info.epoch))
        if swap_mlp:
            all_layer_sizes = [dataset.input_size]+hidden_sizes+[dataset.target_size]
            fwd_ops = [info.sample*d1*d2 for d1, d2 in zip(all_layer_sizes[:-1], all_layer_sizes[1:])]
            back_ops = [info.sample*d1*d2 for d1, d2 in zip(all_layer_sizes[:-1], all_layer_sizes[1:])]
            update_ops = [info.sample*d1*d2 for d1, d2 in zip(all_layer_sizes[:-1], all_layer_sizes[1:])]
        else:
            fwd_ops = [layer_.fwd_op_count.get_value() for layer_ in net.layers]
            back_ops = [layer_.back_op_count.get_value() for layer_ in net.layers]
            update_ops = [layer_.update_op_count.get_value() for layer_ in net.layers]
        if info.epoch!=0:
            with IndentPrint('Mean Ops by epoch {}'.format(info.epoch)):
                print 'Fwd: {}'.format([si_format(ops/info.epoch, format_str='{value} {prefix}Ops') for ops in fwd_ops])
                print 'Back: {}'.format([si_format(ops/info.epoch, format_str='{value} {prefix}Ops') for ops in back_ops])
                print 'Update: {}'.format([si_format(ops/info.epoch, format_str='{value} {prefix}Ops') for ops in update_ops])
        if info.epoch>max(0.5, 2*test_period) and not swap_mlp and score.get_score('train', 'noise_free') < 20:
            raise Exception("This horse ain't goin' nowhere.")

        op_count_info.append((info, (fwd_ops, back_ops, update_ops)))

    info_score_pairs = train_and_test_online_predictor(
        dataset=dataset,
        train_fcn = net.train.compile(),
        predict_fcn=prediction_funcs,
        minibatch_size=minibatch_size,
        n_epochs=n_epochs,
        test_epochs=('every', test_period),
        score_measure='percent_argmax_correct',
        test_on = 'training+test',
        test_callback= test_callback
        )
    return info_score_pairs, op_count_info


with capture_created_experiments() as exps:
    demo_herding_network.add_variant(adaptive_relative_scale=1)
    demo_herding_network.add_variant(adaptive_relative_scale=10)
    X=demo_herding_network.add_variant(swap_mlp=True)
    X.add_variant(learning_rate=0.001)
    demo_herding_network.add_variant(adaptive=False)

    with capture_created_experiments() as exps_grad_calc:
        demo_herding_network.add_variant(grad_calc='true')
        demo_herding_network.add_variant(grad_calc='recon')
        demo_herding_network.add_variant(grad_calc='future')
        demo_herding_network.add_variant(grad_calc='past')
        demo_herding_network.add_variant(grad_calc='past_step')
        demo_herding_network.add_variant(grad_calc='past_matrix')
    for ex in exps_grad_calc:
        ex.add_variant(learning_rate=0.001)
        ex.add_variant(kp=0.5, kd=5.)
        ex.add_variant(kp=0.01, kd=1.)
        ex.add_variant(kp=1., kd=1.)
        ex.add_variant(kp=1., kd=0)
        ex.add_variant(kp=2., kd=0)
        ex.add_variant(kp=.1, kd=.5, learning_rate=0.001)
        ex.add_variant(hidden_sizes=[200, 200])


for exp in exps:
    exp.add_variant(dataset='temporal_mnist')


if __name__ == '__main__':
    browse_experiments()
