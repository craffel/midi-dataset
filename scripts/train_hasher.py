'''
Given a parameter setting, train a cross-modality hasher on the entire dataset.
'''
import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import os
import argparse


def train(params):
    '''
    Wrapper around cross-modality hashing
    '''
    print "Loading data ..."
    # Set up paths
    base_data_directory = '../data'
    hash_data_directory = os.path.join(base_data_directory, 'hash_dataset')
    with open(os.path.join(hash_data_directory, 'train.csv')) as f:
        train_list = f.read().splitlines()
    with open(os.path.join(hash_data_directory, 'valid.csv')) as f:
        valid_list = f.read().splitlines()
    # Load in the data
    (X_train, Y_train, X_validate, Y_validate) = hashing_utils.load_data(
        train_list, valid_list)

    # Compute layer sizes.  Middle layers are nextpow2(input size)
    hidden_layer_size_X = int(2**np.ceil(np.log2(X_train.shape[1])))
    hidden_layer_size_Y = int(2**np.ceil(np.log2(Y_train.shape[1])))
    n_layers = int(params['n_layers'])
    learning_rate = 10**(-params['learning_rate_exp'])
    params = dict([(k, v) for k, v in params.items()
                   if k != 'n_layers' and k != 'learning_rate_exp'])
    print "Training ..."
    epochs = [(epoch, X_params, Y_params) for (epoch, X_params, Y_params)
              in cross_modality_hashing.train_cross_modality_hasher(
                  X_train, Y_train, X_validate, Y_validate,
                  [hidden_layer_size_X]*(n_layers - 1),
                  [hidden_layer_size_Y]*(n_layers - 1),
                  n_bits=16, learning_rate=learning_rate, **params)]
    best_epoch = np.argmin([e[0]['validate_objective'] for e in epochs])
    print "Best objective {}".format(epochs[best_epoch][0])
    return epochs[best_epoch][1], epochs[best_epoch][2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hasher')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--alpha_XY', dest='alpha_XY', type=float, default=.05,
                        help='alpha_XY regularization parameter')
    parser.add_argument('--m_XY', dest='m_XY', type=int, default=5,
                        help='m_XY regularization threshold parameter')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Should we use dropout?')
    parser.add_argument('--learning_rate_exp', dest='learning_rate_exp',
                        type=int, default=4,
                        help='Learning rate negative exponent')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=.999,
                        help='Optimization momentum hyperparameter')
    X_params, Y_params = train(vars(parser.parse_args()))
    if not os.path.exists('../results'):
        os.path.makedirs('../results')
    hashing_utils.save_model(X_params, '../results/model_X.pkl')
    hashing_utils.save_model(Y_params, '../results/model_Y.pkl')
