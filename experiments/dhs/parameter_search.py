'''
Script for running a search over the best network hyperparameters.
'''
import dhs
import numpy as np
import os
import simple_spearmint
import glob
import deepdish
import sys
sys.path.append(os.path.join('..', '..'))
import experiment_utils

RESULTS_PATH = '../../results'


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
    # Construct path to training data
    data_directory = os.path.join(RESULTS_PATH, 'training_dataset')

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
        best_objective, best_epoch, best_model = experiment_utils.run_trial(
            suggestion, data_directory, dhs.train)
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
