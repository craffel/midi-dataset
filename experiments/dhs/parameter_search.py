'''
Script for running a search over the best network hyperparameters.
'''
import dhs
import numpy as np
import os
import simple_spearmint
import glob
import traceback
import deepdish
import collections
import lasagne
import functools
import sys
import utils

RESULTS_PATH = '../../results'


def run_trial(params):
    # Load in data as dictionary of dictionaries
    data = {'train': collections.defaultdict(list),
            'valid': collections.defaultdict(list)}
    data_directory = os.path.join(RESULTS_PATH, 'training_dataset')
    for set in ['train', 'valid']:
        for f in glob.glob(os.path.join(data_directory, set, 'h5', '*.h5')):
            for k, v in deepdish.io.load(f).items():
                data[set][k].append(v)

    # Build networks
    layers = {}
    for network in ['X', 'Y']:
        # Get # of features (last dimension) from first training sequence
        input_shape = (None, 1, None, data['train'][network][0].shape[-1])
        # Get training set statistics for standardization
        input_mean = np.mean(
            np.concatenate(data['train'][network], axis=1), axis=1)
        input_std = np.std(
            np.concatenate(data['train'][network], axis=1), axis=1)
        # Choose network structure based on network param
        if params['network'] == 'big_filter':
            build_network = utils.build_network_big_filter
        elif params['network'] == 'small_filters':
            build_network = utils.build_network_small_filters
        else:
            raise ValueError('Unknown network {}'.format(params['network']))
        layers[network] = build_network(
            input_shape, input_mean, input_std,
            params['downsample_frequency'], params['dropout'])

    # Create updates-creating function
    updates_function = functools.partial(
        lasagne.updates.rmsprop, learning_rate=params['learning_rate'],
        rho=params['momentum'])

    print ',\n'.join(['\t{} : {}'.format(k, v) for k, v in params.items()])
    # Create a list of epochs
    epochs = []
    # Keep track of lowest objective found so far
    best_objective = np.inf
    try:
        for epoch in dhs.train(
                data['train']['X'], data['train']['Y'], data['valid']['X'],
                data['valid']['Y'], layers, params['negative_importance'],
                params['negative_threshold'], params['entropy_importance'],
                updates_function):
            # Stop training if a nan training cost is encountered
            if not np.isfinite(epoch['train_cost']):
                break
            epochs.append(epoch)
            if epoch['validate_objective'] < best_objective:
                best_objective = epoch['validate_objective']
                best_epoch = epoch
                best_model = {
                    'X': lasagne.layers.get_all_param_values(layers['X']),
                    'Y': lasagne.layers.get_all_param_values(layers['Y'])}
            print "{}: {}, ".format(epoch['iteration'],
                                    epoch['validate_objective']),
            sys.stdout.flush()
    # If there was an error while training, report it to whetlab
    except Exception:
        print "ERROR: "
        print traceback.format_exc()
        return np.nan, {}, {}
    print
    # Check that all training costs were not NaN; return NaN if any were.
    success = np.all([np.isfinite(e['train_cost']) for e in epochs])
    if np.isinf(best_objective) or len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return np.nan, {}, {}
    else:
        for k, v in best_epoch.items():
            print "\t{:>35} | {}".format(k, v)
        print
        return best_objective, best_epoch, best_model


if __name__ == '__main__':
    # Define hyperparameter space
    space = {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'negative_threshold': {'type': 'int', 'min': 1, 'max': 16},
        'dropout': {'type': 'int', 'min': 0, 'max': 1},
        'learning_rate': {'type': 'float', 'min': .0001, 'max': .01},
        'negative_importance': {'type': 'float', 'min': 0.01, 'max': 1.},
        'entropy_importance': {'type': 'float', 'min': 0.0, 'max': 1.},
        'downsample_frequency': {'type': 'int', 'min': 0, 'max': 1},
        'network': {'type': 'enum', 'options': ['big_filter', 'small_filters']}
    }

    # Create parameter trials directory if it doesn't exist
    trial_directory = os.path.join(RESULTS_PATH, 'dhs_parameter_trials')
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)
    # Same for model directory
    model_directory = os.path.join(RESULTS_PATH, 'dhs_model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Create SimpleSpearmint suggester instance
    ss = simple_spearmint.SimpleSpearmint(space)

    # Load in previous results for "warm start"
    for trial_file in glob.glob(os.path.join(trial_directory, '*.h5')):
        trial = deepdish.io.load(trial_file)
        ss.update(trial['hyperparameters'], trial['best_objective'])

    # Run parameter optimization forever
    while True:
        # Get a new suggestion
        suggestion = ss.suggest()
        # Train a network with these hyperparameters
        best_objective, best_epoch, best_model = run_trial(suggestion)
        # Update spearmint on the result
        ss.update(suggestion, best_objective)
        # Write out a result file
        trial_filename = ','.join('{}={}'.format(k, v)
                                  for k, v in suggestion.items()) + '.h5'
        deepdish.io.save(
            os.path.join(trial_directory, trial_filename),
            {'hyperparameters': suggestion, 'best_objective': best_objective,
             'best_epoch': best_epoch})
        # Also write out the entire model when the objective is the smallest
        # We don't want to write all models; they are > 100MB each
        if (not np.isnan(best_objective) and
                best_objective == np.min(ss.objective_values)):
            deepdish.io.save(
                os.path.join(model_directory, 'best_model.h5'), best_model)
