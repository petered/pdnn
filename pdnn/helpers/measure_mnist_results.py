from collections import OrderedDict
import itertools

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

from artemis.general.should_be_builtins import remove_duplicates, all_equal, try_key
from artemis.plotting.expanding_subplots import select_subplot
from pdnn.helpers.forward_pass import forward_pass_and_cost
from pdnn.helpers.sparse_td_net import tdnet_forward_pass_cost_and_output
from pdnn.helpers.temporal_mnist import get_temporal_mnist_dataset
from artemis.ml.predictors.train_and_test import percent_argmax_incorrect
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.general.progress_indicator import ProgressIndicator


__author__ = 'peter'


def get_mnist_results_with_parameters(weights, biases, scales = None, hidden_activations='relu', output_activation='softmax', n_samples = None, smoothing_steps = 1000):
    """
    Return a data structure showing the error and computation for required by the orignal, rounding, and sigma-delta
    implementation of a network with the given parameters.

    :param weights:
    :param biases:
    :param scales:
    :param hidden_activations:
    :param output_activation:
    :param n_samples:
    :param smoothing_steps:
    :return: results: An OrderedDict
        Where the key is a 3-tuple are:
            (dataset_name, subset, net_version), Where:
                dataset_name: is 'mnist' or 'temp_mnist'
                subset: is 'train' or 'test'
                net_version: is 'td' or 'round' or 'truth'
        And values are another OrderedDict, with keys:
            'MFlops', 'l1_errorm', 'class_error'  ... for discrete nets and
            'Dense MFlops', 'Sparse MFlops', 'class_error' for "true" nets.
    """
    mnist = get_mnist_dataset(flat=True, n_training_samples=n_samples, n_test_samples=n_samples)
    temp_mnist = get_temporal_mnist_dataset(flat=True, smoothing_steps=smoothing_steps, n_training_samples=n_samples, n_test_samples=n_samples)
    results = OrderedDict()
    p = ProgressIndicator(2*3*2)
    for dataset_name, (tr_x, tr_y, ts_x, ts_y) in [('mnist', mnist.xyxy), ('temp_mnist', temp_mnist.xyxy)]:
        for subset, x, y in [('train', tr_x, tr_y), ('test', ts_x, ts_y)]:
            traditional_net_output, dense_flops, sparse_flops = forward_pass_and_cost(
                input_data = x,
                weights=weights,
                biases=biases,
                hidden_activations=hidden_activations,
                output_activations=output_activation
                )
            assert round(dense_flops)==dense_flops and round(sparse_flops)==sparse_flops, 'Flop counts must be int!'

            class_error = percent_argmax_incorrect(traditional_net_output, y)
            results[dataset_name, subset, 'truth'] = OrderedDict([('Dense MFlops', dense_flops/(1e6*len(x))), ('Sparse MFlops', sparse_flops/(1e6*len(x))), ('class_error', class_error)])
            for net_version in 'td', 'round':
                (comp_cost_adds, comp_cost_multiplyadds), output = tdnet_forward_pass_cost_and_output(
                    inputs=x,
                    weights= weights,
                    biases = biases,
                    scales = scales,
                    version=net_version,
                    hidden_activations = hidden_activations,
                    output_activations = output_activation,
                    quantization_method='herd',
                    computation_calc=('adds', 'multiplyadds')
                    )
                l1_error = np.abs(output-traditional_net_output).sum(axis=1).mean(axis=0)
                class_error = percent_argmax_incorrect(output, y)
                results[dataset_name, subset, net_version] = OrderedDict([
                    ('MFlops', comp_cost_adds/(1e6*len(x))),
                    ('MFlops-multadd', comp_cost_multiplyadds/(1e6*len(x))),
                    ('l1_error', l1_error),
                    ('class_error', class_error)
                    ])
                p.print_update()
    return results


def get_data_structure_info(result_struct):

    all_settings_ = result_struct.keys()
    first_result = result_struct.values()[0]
    all_measurements_ = remove_duplicates([k for v in first_result.values() for k in v.keys()])  # ['flops', 'error', ...]
    all_datasets_, all_subsets_, all_nets_ = [remove_duplicates(all_category_entries) for all_category_entries in zip(*[categories for categories in first_result.keys()])]
    return all_settings_, all_measurements_, all_datasets_, all_subsets_, all_nets_


def display_discrete_network_results(results, cleanup=True, clear_repeated_headers = True, tablefmt='latex'):
    """
    Ok, lets try this one more time and lets try not to reimplement relational databases.

    Display a table containing the results of an experiment.

    :param results: An object of the form
        OrderedDict<
            setting -> OrderedDict<
                (dataset_name, subset_name, net_name) -> OrderedDict<
                    measurement_name -> measurement_value
                    >
                >
            >
        Where:
            setting is a descrition of the setup (eg 'unoptimized', 'lambda=0.001')
            (dataset_name, subset_name, net_name) is for example(eg ('mnist', 'test', 'td'))
            measurement_name is a string identifying the measurement (eg 'Flops', 'error')
            measurement_value is a value for the associated measurement (eg 4325535, 0.424)
    """

    plot_mnist_energy_results(results)



    if cleanup:
        net_name_change = OrderedDict([('truth', 'Original'), ('round', 'Round'), ('td', '$\Sigma\Delta$')])
        new_results = OrderedDict()
        all_settings, all_measurements, all_datasets, all_subsets, all_nets = get_data_structure_info(results)
        for setting_name, result in results.iteritems():
            current_new_result = OrderedDict()
            for dataset_name, subset_name, net_name in itertools.product(all_datasets, all_subsets, all_nets):

                old_measures = result[dataset_name, subset_name, net_name]
                new_net_name = {'truth': 'Original', 'round': 'Round', 'td': '$\Sigma\Delta$'}[net_name]
                current_new_result[dataset_name, new_net_name] = OrderedDict()

                # if net_name=='truth':
                #     result[dataset_name, subset_name, 'truth']['MFlops'] = '%d \\ %d' % (result[dataset_name, subset_name, 'truth']['Dense MFlops'], result[dataset_name, subset_name, 'truth']['Sparse MFlops'])
                # measures = result[dataset_name, subset_name, net_name]


                if subset_name=='test':
                    current_new_result[dataset_name, new_net_name]['kFlops Test (ds\\sp)'] = \
                        '%d \\ %d' % (old_measures['Dense MFlops']*1e3, old_measures['Sparse MFlops']*1e3) if net_name=='truth' else \
                        '%d' % (old_measures['MFlops']*1e3, )
                    current_new_result[dataset_name, new_net_name]['class_error (tr\\ts)'] = \
                        '%.3g \\ %.3g' % (result[dataset_name, 'train', net_name]['class_error'], result[dataset_name, 'test', net_name]['class_error'])

                    current_new_result[dataset_name, new_net_name]['Int32-Energy (nJ)'] = \
                        '%.3g \\ %.3g' % (
                            estimate_energy_cost(old_measures['Dense MFlops'], op='mult-add', dtype='int', n_bits=32)*1e3,   # pico*mega=micro
                            estimate_energy_cost(old_measures['Sparse MFlops'], op='mult-add', dtype='int', n_bits=32)*1e3
                            ) if net_name=='truth' else \
                        '%.3g' % (estimate_energy_cost(old_measures['MFlops'], op='add', dtype='int', n_bits=32)*1e3)

            new_results[setting_name.replace('lambda', '$\lambda$')] = current_new_result

        rows = build_table(
            lambda (_setting, _net), (_ds, _meas): new_results[_setting][_ds, _net][_meas],
            row_categories=[new_results.keys(), net_name_change.values()],
            row_header_labels = ['Setting', 'Net Type'],
            column_categories = (all_datasets, new_results.values()[0].values()[0].keys()),
            clear_repeated_headers = clear_repeated_headers
            )

        rows.insert(2, ["="*10]*len(rows[0]))

        tab_data = tabulate(rows, floatfmt='%06.2f', tablefmt=tablefmt)

        if tablefmt == 'latex':
            # tab_data=tab_data.replace('\$', '$').replace('\\textbackslash{}', '\\') #.replace('\$\\textbackslash{}lambda\$', '$\lambda$')
            # tab_data=tab_data.replace('\$\\textbackslash{}lambda\$', '$\\lambda$')#.replace('\$', '$').replace('\\textbackslash{}', '\\') #.replace('\$\\textbackslash{}lambda\$', '$\lambda$')

            tab_data = tab_data.replace('\$\\textbackslash{}', '$\\').replace('\$', '$').replace('\\textbackslash{}Delta', '\\Delta')

        print tab_data


    else:
        # Get all parameters of the data object.
        all_settings, all_measurements, all_datasets, all_subsets, all_nets = get_data_structure_info(results.values()[0])

        # Define how the data object will be mapped into a table
        def get_data_at(row_info_, column_info_):
            setting_name, net_type = row_info_
            dataset_name, measurement_name, subset_name = column_info_
            return try_key(results[setting_name][dataset_name, subset_name, net_type], measurement_name, ' ')

        rows = build_table(get_data_at,
            row_categories=(all_settings, all_nets),
            row_header_labels = ['Setting', 'Net Type'],
            column_categories = (all_datasets, all_measurements, all_subsets),
            clear_repeated_headers = clear_repeated_headers
            )

        rows.insert(3, ["="*10]*len(rows[0]))

        print tabulate(rows, floatfmt='%06.2f')


def estimate_energy_cost(n_ops, op, dtype, n_bits=32):
    """
    Return the estimate cost in pJ for n_ops operations.
    We use numbers from Horowitz paper: http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=6757323

    :param n_ops: Number of operations
    :param op: 'add' or 'mult'
    :param dtype: 'float' or 'int'
    :param n_bits: Number of bits (8, 16, 32)
    :return: Estimated cost of your operation, in pJ
    """
    # Numbers are from: http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=6757323

    if isinstance(n_ops, (list, tuple)):
        n_ops = np.array(n_ops)

    costs = {
        ('int', 'add', 8): 0.03,
        ('int', 'add', 32): 0.1,
        ('int', 'mult', 8): 0.2,
        ('int', 'mult', 32): 3.1,
        ('float', 'add', 16): 0.4,
        ('float', 'add', 32): 0.9,
        ('float', 'mult', 16): 1.1,
        ('float', 'mult', 32): 3.7,
        }

    if op=='mult-add':
        # Note, we consider the n_ops to be (n_multiplies + n_adds), and assume that n_multiplies==n_ads
        # (this is more or less true if we're doing big dot-products)
        return costs[dtype, 'mult', n_bits]*n_ops/2. + costs[dtype, 'add', n_bits]*n_ops/2.
    else:
        return costs[dtype, op, n_bits]*n_ops


def plot_mnist_energy_results(results, x_scale = ('flops', 'int-energy')):

    # all_settings_, all_measurements_, all_datasets_, all_subsets_, all_nets_ = get_data_structure_info(results)

    if isinstance(x_scale, basestring):
        x_scale = [x_scale]
    assert all(x in ('flops', 'multadd-flops', 'int-energy', 'float-energy') for x in x_scale)

    subset = 'test'

    lambda_results = [res for name, res in results.iteritems() if 'lambda' in name.lower()]

    dataset_names = OrderedDict([('mnist', 'MNIST'), ('temp_mnist', 'Temporal MNIST')])

    plt.figure()
    first_plot=True
    for x_s in x_scale:
        for dataset in dataset_names:
            last_plot = x_s==x_scale[-1] and dataset==dataset_names.keys()[-1]
            ax=select_subplot((x_s, dataset), layout='h')
            plt.subplots_adjust(bottom=.15, wspace=0)

            flop_measure = 'MFlops' if x_s=='flops' else 'MFlops-multadd'

            td_flops = [res[dataset, subset, 'td'][flop_measure] for res in lambda_results]
            td_errors = [res[dataset, subset, 'td']['class_error'] for res in lambda_results]
            round_flops = [res[dataset, subset, 'round'][flop_measure] for res in lambda_results]
            round_errors = [res[dataset, subset, 'round']['class_error'] for res in lambda_results]
            original_flops = results['unoptimized'][dataset, subset, 'truth']['Sparse MFlops'], results['unoptimized'][dataset, subset, 'truth']['Dense MFlops']
            original_errors = [results['unoptimized'][dataset, subset, 'truth']['class_error']]*2
            if x_s in ('int-energy', 'float-energy'):
                dtype = 'int' if x_s=='int-energy' else 'float'
                # Cet the energy in nJ.  (pJ/Op)*(N MPp)->(muJ/Op).  (muJ/Op)*(1e3 n/mu) -> nJ/Op
                td_x = estimate_energy_cost(td_flops, 'add', dtype=dtype, n_bits=32)*1e3
                round_x = estimate_energy_cost(round_flops, 'add', dtype=dtype, n_bits=32)*1e3
                original_x = estimate_energy_cost(original_flops, 'mult-add', dtype=dtype, n_bits=32)*1e3

            else:
                td_x, round_x, original_x = np.array(td_flops)*1e3, np.array(round_flops)*1e3, np.array(original_flops)*1e3
            plt.plot(round_x, round_errors, linewidth=2, label='Rounding Network', marker='.', markersize=10)
            plt.plot(td_x, td_errors, label='$\Sigma\Delta$ Network', linewidth=2, marker='.', markersize=10)
            plt.plot(original_x, original_errors, label = 'Original Network', marker='.', markersize=20, linewidth=2)

            if x_s in ('int-energy', 'float-energy'):
                plt.xlabel('nJ/sample')  # pico*mega = micro
                plt.gca().set_xscale('log')
            else:
                plt.xlabel('kOps/sample')
                for label in ax.get_xticklabels()[::2]:
                    label.set_visible(False)
            plt.gca().set_yscale('log')
            plt.title(dataset_names[dataset])
            plt.grid()
            yticks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
            plt.gca().yaxis.set_ticks(yticks)
            plt.gca().set_ylim(1.5, 20)
            if first_plot:
                plt.ylabel('Classification Error (%)')
                first_plot=False
                plt.gca().set_yticklabels([str(i) for i in yticks])
            else:
                plt.gca().set_yticklabels(['' for _ in yticks])
            if last_plot:
                plt.legend(loc='upper right', framealpha=0.5, fontsize='medium')
    plt.show()



def build_table(lookup_fcn, row_categories, column_categories, clear_repeated_headers = True, prettify_labels = True,
            row_header_labels = None):
    """
    Build the rows of a table.  You can feed these rows into tabulate to generate pretty things.

    :param lookup_fcn: A function of the form:
        data = lookup_fcn(row_info, column_info)
        Where:
            row_info is a tuple of data identifying the row.
            col_info is a tuple of data identifying the column
    :param row_categories: A list<list<str>> of categories that will make up the rows
    :param column_categories: A list<list<str>> of catefories that will make up the columns
    :param clear_repeated_headers: True to not repeat row headers.
    :param row_header_labels: Labels for the row headers.
    :return: A list of rows.
    """
    # Now, build that table!
    if row_header_labels is not None:
        assert len(row_header_labels) == len(row_categories)
    rows = []
    column_headers = zip(*itertools.product(*column_categories))
    for i, c in enumerate(column_headers):
        row_header = row_header_labels if row_header_labels is not None and i==len(column_headers)-1 else [' ']*len(row_header_labels)
        row = row_header+blank_out_repeats(c) if clear_repeated_headers else list(c)
        rows.append([prettify_label(el) for el in row] if prettify_labels else row)
    last_row_data = [' ']*len(row_categories)
    for row_info in itertools.product(*row_categories):
        if blank_out_repeats:
            row_header, last_row_data = zip(*[(h, h) if lh!=h else (' ', lh) for h, lh in zip(row_info, last_row_data)])
        else:
            row_header = row_info
        if prettify_labels:
            row_header = [prettify_label(str(el)) for el in row_header]
        data = [lookup_fcn(row_info, column_info) for column_info in itertools.product(*column_categories)]
        rows.append(list(row_header) + data)
    assert all_equal(len(r) for r in rows)
    return rows
    # print tabulate(rows)


def prettify_label(label):
    return (label[0].upper() + label[1:]).replace('_', ' ')


def blank_out_repeats(sequence, replace_with=' '):

    new_sequence = list(sequence)
    for i in xrange(len(new_sequence)-1, 0, -1):
        if new_sequence[i]==new_sequence[i-1]:
            new_sequence[i] = replace_with
    return new_sequence


def filter_results(results, category_filters):

    assert all_equal(len(k) for k in results.keys())
    assert len(results.keys()[0])==len(category_filters)
    filtered_results = OrderedDict((
            tuple(cv for cv, cf in zip(category_values, category_filters) if isinstance(cf, list)),
            measures
            ) for category_values, measures in zip(results)
                if all(cv in cf if isinstance(cf, list) else cv==cf for cv, cf in zip(category_values, category_filters))
        )
    return filtered_results



# def plot_results(results, x_measure, y_measure, columns_to_merge):










