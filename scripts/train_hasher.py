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

    # Use the # of hidden layers and the hidden layer power to construct a list
    # [2^hidden_power, 2^hidden_power, ...n_hidden times...]
    hidden_layer_sizes = [2**params['hidden_pow']]*params['n_hidden']
    params['hidden_layer_sizes'] = {'X': hidden_layer_sizes,
                                    'Y': hidden_layer_sizes}
    # Use the number of convolutional layers to construct a list
    # [16, 32 ...n_conv times]
    num_filters = [2**(n + 4) for n in xrange(params['n_conv'])]
    params['num_filters'] = {'X': num_filters, 'Y': num_filters}
    # For X modality, first filter is 12 semitones tall.  For Y modality, that
    # would squash the entire dimension, so the first filter is 3 tall
    params['filter_size'] = {'X': [(5, 12)] + [(3, 3)]*(params['n_conv'] - 1),
                             'Y': [(5, 3)] + [(3, 3)]*(params['n_conv'] - 1)}
    # Construct a downsample list [(1, 2), (1, 2), ...n_conv times...]
    ds = [(1, 2)]*params['n_conv']
    params['ds'] = {'X': ds, 'Y': ds}
    # Remove hidden_pow, n_hidden, and n_conv parameters
    params = dict([(k, v) for k, v in params.items()
                   if k != 'hidden_pow' and k != 'n_hidden' and k != 'n_conv'])
    print "Training ..."
    epochs = [(epoch, X_params, Y_params) for (epoch, X_params, Y_params)
              in cross_modality_hashing.train_cross_modality_hasher(
                  X_train, Y_train, X_validate, Y_validate, **params)]
    best_epoch = np.argmax([e[0]['validate_objective'] for e in epochs])
    print "Best objective {}".format(epochs[best_epoch][0])
    return epochs[best_epoch][1], epochs[best_epoch][2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hasher')
    parser.add_argument('--alpha_XY', dest='alpha_XY', type=float, default=.2,
                        help='alpha_XY regularization parameter')
    parser.add_argument('--alpha_stress', dest='alpha_stress', type=float,
                        default=.1, help='Stress regularization parameter')
    parser.add_argument('--dropout', dest='dropout', type=bool, default=False,
                        help='Should we use dropout?')
    parser.add_argument('--hidden_pow', dest='hidden_pow', type=int,
                        default=11, help='Hidden layer size exponent')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=.001, help='Learning rate')
    parser.add_argument('--m_XY', dest='m_XY', type=int, default=3,
                        help='m_XY regularization threshold parameter')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=0.5, help='Optimization momentum')
    parser.add_argument('--n_conv', dest='n_conv', type=int, default=2,
                        help='Number of convolutional layers')
    parser.add_argument('--n_hidden', dest='n_hidden', type=int, default=2,
                        help='Number of hidden layers')
    X_params, Y_params = train(vars(parser.parse_args()))
    if not os.path.exists('../results'):
        os.path.makedirs('../results')
    hashing_utils.save_model(X_params, '../results/model_X.pkl')
    hashing_utils.save_model(Y_params, '../results/model_Y.pkl')
