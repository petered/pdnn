from artemis.experiments.experiment_record import run_experiment, has_experiment_record, load_lastest_experiment_results
from pdnn.helpers.demo_pdnn_mnist import do_the_figure
from pdnn.helpers.demo_kd_too_large import demo_kd_too_large
from pdnn.helpers.demo_pd_stdp_equivalence import demo_pd_stdp_equivalence, demo_plot_stdp
from pdnn.helpers.demo_visualize_k_effects import demo_visualize_k_effects
from pdnn.helpers.demo_weight_update_figures import demo_weight_update_figures
from pdnn.helpers.demo_why_kp_explanation import demo_why_kp_explanation
from pdnn.helpers.demo_temporal_mnist import demo_plot_temporal_mnist


def generate_figure(figure_number):
    if figure_number==1:
        demo_visualize_k_effects()
    elif figure_number==2:
        demo_why_kp_explanation()
    elif figure_number==3:
        demo_weight_update_figures()
    elif figure_number==4:
        demo_plot_stdp()
    elif figure_number==5:
        # Do MNIST Experiment and plot results... this will take several hours on the first run, but will be fast after that.
        mlp_exp = 'demo_herding_network.swap_mlp=True'
        pdnn_exp = 'demo_herding_network.grad_calc=past_matrix'
        tmlp_exp = 'demo_herding_network.swap_mlp=True.dataset=temporal_mnist'
        tpdnn_exp = 'demo_herding_network.grad_calc=past_matrix.dataset=temporal_mnist'
        for experiment_id in (mlp_exp, pdnn_exp, tmlp_exp, tpdnn_exp):
            if not has_experiment_record(experiment_id):
                run_experiment(experiment_id)
        do_the_figure(
            mnist_results=load_lastest_experiment_results([mlp_exp, pdnn_exp]),
            temporal_mnist_results=load_lastest_experiment_results([tmlp_exp, tpdnn_exp]),
            names={mlp_exp: 'MLP',pdnn_exp: 'PDNN',tmlp_exp: 'MLP',tpdnn_exp: 'PDNN'}
            )
    elif figure_number==6:
        demo_plot_temporal_mnist()
    elif figure_number==7:
        demo_kd_too_large().display_or_run()
    elif figure_number==8:
        demo_pd_stdp_equivalence()
    else:
        raise Exception('No figure {}'.format(figure_number))


if __name__ == '__main__':
    figure_number = raw_input('Select which figure you want to reproduce (1-8) and press enter.\n'
                              'Figures (5 and 7) will take a long time to generate the first \n'
                              'time, but after that the results will be saved and they will \n'
                              'display quickly (note that on some systems the figure may pop-up\n'
                              'in the background).  >>')
    generate_figure(int(figure_number))
