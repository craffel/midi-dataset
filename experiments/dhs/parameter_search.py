'''
Script for running a search over the best network hyperparameters.
'''
import dhs
import os
import sys
sys.path.append(os.path.join('..', '..'))
import experiment_utils

RESULTS_PATH = '../../results'


if __name__ == '__main__':
    # Define hyperparameter space
    space = {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'negative_threshold': {'type': 'int', 'min': 1, 'max': 16},
        'learning_rate': {'type': 'float', 'min': .000001, 'max': .01},
        'negative_importance': {'type': 'float', 'min': 0.01, 'max': 20.},
        'output_l2_penalty': {'type': 'float', 'min': 0.0, 'max': 1.},
        'downsample_frequency': {'type': 'int', 'min': 0, 'max': 1},
        'network': {'type': 'enum',
                    'options': ['dhs_big_filter', 'dhs_small_filters']}
    }

    # Construct paths
    trial_directory = os.path.join(RESULTS_PATH, 'dhs_parameter_trials')
    model_directory = os.path.join(RESULTS_PATH, 'dhs_model')
    data_directory = os.path.join(RESULTS_PATH, 'training_dataset')

    # Run parameter optimization forever
    experiment_utils.parameter_search(
        space, trial_directory, model_directory, data_directory, dhs.train)
