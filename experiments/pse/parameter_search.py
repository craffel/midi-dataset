""" Search for good hyperparameters for embedding sequences of audio and MIDI
spectrograms into a common space. """
import pse
import os
import sys
sys.path.append(os.path.join('..', '..'))
import experiment_utils

RESULTS_PATH = '../../results'


if __name__ == '__main__':
    # Define hyperparameter space
    space = {
        'momentum': {'type': 'float', 'min': 0., 'max': 0.999},
        'negative_threshold': {'type': 'float', 'min': .01, 'max': 10},
        'learning_rate': {'type': 'float', 'min': 1e-6, 'max': 1e-4},
        'negative_importance': {'type': 'float', 'min': 0.01, 'max': 100.},
        'n_conv': {'type': 'int', 'min': 0, 'max': 3},
        'n_attention': {'type': 'int', 'min': 1, 'max': 4},
        'network': {'type': 'enum',
                    'options': ['pse_big_filter', 'pse_small_filters']}
    }

    # Construct paths
    trial_directory = os.path.join(RESULTS_PATH, 'pse_parameter_trials')
    model_directory = os.path.join(RESULTS_PATH, 'pse_model')
    data_directory = os.path.join(RESULTS_PATH, 'training_dataset_unaligned')

    # Run parameter optimization forever
    experiment_utils.parameter_search(
        space, trial_directory, model_directory, data_directory, pse.train)
