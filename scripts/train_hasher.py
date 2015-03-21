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
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, n_bits)


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

    learning_rate = 10**(-params['learning_rate_exp'])
    params = dict([(k, v) for k, v in params.items()
                   if k != 'learning_rate_exp'])
    print "Training ..."
    epochs = [(epoch, X_params, Y_params) for (epoch, X_params, Y_params)
              in cross_modality_hashing.train_cross_modality_hasher(
                  X_train, Y_train, X_validate, Y_validate, num_filters,
                  filter_size, ds, hidden_layer_sizes, n_bits=n_bits,
                  learning_rate=learning_rate, **params)]
    best_epoch = np.argmin([e[0]['validate_objective'] for e in epochs])
    print "Best objective {}".format(epochs[best_epoch][0])
    return epochs[best_epoch][1], epochs[best_epoch][2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hasher')
    parser.add_argument('--alpha_XY', dest='alpha_XY', type=float, default=.2,
                        help='alpha_XY regularization parameter')
    parser.add_argument('--m_XY', dest='m_XY', type=int, default=4,
                        help='m_XY regularization threshold parameter')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Should we use dropout?')
    parser.add_argument('--learning_rate_exp', dest='learning_rate_exp',
                        type=int, default=3,
                        help='Learning rate negative exponent')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=0,
                        help='Optimization momentum hyperparameter')
    X_params, Y_params = train(vars(parser.parse_args()))
    if not os.path.exists('../results'):
        os.path.makedirs('../results')
    hashing_utils.save_model(X_params, '../results/model_X.pkl')
    hashing_utils.save_model(Y_params, '../results/model_Y.pkl')
