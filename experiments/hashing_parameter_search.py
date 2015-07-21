import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import os
import spearmint.main
import glob
import traceback

BASE_DATA_DIRECTORY = '../data'


def main(job_id, params):
    # Spearmint requires (I think) all params to be passed as at least
    # 1-dimensional arrays.  So, get the first entry to flatten.
    for key, value in params.items():
        params[key] = value[0]
        if np.isscalar(params[key]):
            params[key] = params[key].item()

    hash_data_directory = os.path.join(BASE_DATA_DIRECTORY, 'hash_dataset')
    train_list = list(glob.glob(os.path.join(
        hash_data_directory, 'train', 'npz', '*.npz')))
    valid_list = list(glob.glob(os.path.join(
        hash_data_directory, 'valid', 'npz', '*.npz')))
    # Load in the data
    (X_train, Y_train, X_validate, Y_validate) = hashing_utils.load_data(
        train_list, valid_list)

    # Use the # of hidden layers and the hidden layer power to construct a list
    # [2^hidden_power, 2^hidden_power, ...n_hidden times...]
    hidden_layer_sizes = [2**11]*params['n_hidden']
    params['hidden_layer_sizes'] = {'X': hidden_layer_sizes,
                                    'Y': hidden_layer_sizes}
    # Use the number of convolutional layers to construct a list
    # [16, 32 ...n_conv times]
    num_filters = [2**(n + 4) for n in xrange(params['n_conv'])]
    params['num_filters'] = {'X': num_filters, 'Y': num_filters}
    # First filter is 12 semitones tall
    params['filter_size'] = {'X': [(5, 12)] + [(3, 3)]*(params['n_conv'] - 1),
                            'Y': [(5, 12)] + [(3, 3)]*(params['n_conv'] - 1)}
    # Construct a downsample list [(2, 2), (2, 2), (1, 2) ...n_conv-2 times...]
    ds = [(2, 2), (2, 2)] + [(1, 2)]*(params['n_conv'] - 2)
    params['ds'] = {'X': ds, 'Y': ds}
    # Remove hidden_pow, n_hidden, and n_conv parameters
    params = dict([(k, v) for k, v in params.items()
                if k != 'hidden_pow' and k != 'n_hidden' and k != 'n_conv'])
    for k, v in params.items():
        print '\t{} : {},'.format(k, v)
    # Train hasher
    epochs = []
    try:
        for epoch, _, _ in cross_modality_hashing.train_cross_modality_hasher(
                X_train, Y_train, X_validate, Y_validate, **params):
            # Stop training of a nan training cost is encountered
            if not np.isfinite(epoch['train_cost']):
                break
            epochs.append(epoch)
            print "{}: {}, ".format(epoch['iteration'],
                                    epoch['validate_objective']),
            sys.stdout.flush()
    # If there was an error while training, report it to whetlab
    except Exception:
        print "ERROR: "
        print traceback.format_exc()
        return np.nan
    print
    # Check that all training costs were not NaN; return NaN if any were.
    success = np.all([np.isfinite(e['train_cost']) for e in epochs])
    if len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return np.nan
    else:
        # If training was successful, find the maximum validation objective
        best_objective = np.max([e['validate_objective'] for e in epochs])
        best_epoch = [e for e in epochs
                    if e['validate_objective'] == best_objective][0]
        for k, v in best_epoch.items():
            print "\t{:>35} | {}".format(k, v)
        print
        # Negate it, as we will be minimizing the objective
        return -best_objective


if __name__ == '__main__':
    # Define hyperparameter space
    space = {
        'n_hidden': {'type': 'INT', 'size': 1, 'min': 2, 'max': 3.},
        'momentum': {'type': 'FLOAT', 'size': 1, 'min': 0, 'max': 1.},
        'm_XY': {'type': 'INT', 'size': 1, 'min': 0, 'max': 8},
        'dropout': {'type': 'INT', 'size': 1, 'min': 0, 'max': 1},
        'learning_rate': {'type': 'FLOAT', 'size': 1,
                          'min': .0001, 'max': .01},
        'alpha_XY': {'type': 'FLOAT', 'size': 1, 'min': 0., 'max': 1.},
        'n_conv': {'type': 'INT', 'size': 1, 'min': 2, 'max': 3}}

    # Set up spearmint options dict
    options = {'language': 'PYTHON',
               'main-file': os.path.basename(__file__),
               'experiment-name': 'hashing_parameter_search',
               'likelihood': 'GAUSSIAN',
               'variables': space}

    spearmint.main.main(options, os.getcwd())
