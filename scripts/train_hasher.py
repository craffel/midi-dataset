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
import glob


def train(params):
    '''
    Wrapper around cross-modality hashing
    '''
    print "Loading data ..."
    # Set up paths
    base_data_directory = '../data'
    hash_data_directory = os.path.join(base_data_directory, 'hash_dataset')
    train_list = list(glob.glob(os.path.join(
        hash_data_directory, 'train', 'npz', '*.npz')))
    valid_list = list(glob.glob(os.path.join(
        hash_data_directory, 'valid', 'npz', '*.npz')))
    # Load in the data
    (X_train, Y_train, X_validate, Y_validate) = hashing_utils.load_data(
        train_list, valid_list)
    # TODO: Hack to use old convolutional hashing data
    if X_train[0].ndim > 2:
        X_train = [x[0] for x in X_train]
        Y_train = [x[0] for x in Y_train]
        X_validate = [x[0] for x in X_validate]
        Y_validate = [x[0] for x in Y_validate]
    # Compute max length as median of lengths
    max_length_X = int(np.median([len(X) for X in X_train + X_validate]))
    max_length_Y = int(np.median([len(Y) for Y in Y_train + Y_validate]))
    # Turn into sequence matrices/masks
    (X_train, X_train_mask, X_validate,
     X_validate_mask) = hashing_utils.stack_sequences(
         max_length_X, X_train, X_validate)
    (Y_train, Y_train_mask, Y_validate,
     Y_validate_mask) = hashing_utils.stack_sequences(
         max_length_Y, Y_train, Y_validate)

    # Compute layer sizes.  Middle layers are nextpow2(input size)
    hidden_layer_size_X = int(2**np.ceil(np.log2(X_train.shape[2])))
    hidden_layer_size_Y = int(2**np.ceil(np.log2(Y_train.shape[2])))
    n_layers = int(params['n_layers'])
    params['hidden_layer_sizes'] = {
        'X': [hidden_layer_size_X]*(n_layers - 1),
        'Y': [hidden_layer_size_Y]*(n_layers - 1)}
    params['learning_rate'] = 10**(-params['learning_rate_exp'])
    params = dict([(k, v) for k, v in params.items()
                   if k != 'n_layers' and k != 'learning_rate_exp'])
    print "Training ..."
    epochs = []
    for epoch in cross_modality_hashing.train_cross_modality_hasher(
            X_train, X_train_mask, Y_train, Y_train_mask, X_validate,
            X_validate_mask, Y_validate, Y_validate_mask, **params):
        print epoch[0]
        epochs.append(epoch)

    best_epoch = np.argmin([e[0]['validate_objective'] for e in epochs])
    print "Best objective {}".format(epochs[best_epoch][0])
    return epochs[best_epoch][1], epochs[best_epoch][2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hasher')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--alpha_XY', dest='alpha_XY', type=float, default=.05,
                        help='alpha_XY regularization parameter')
    parser.add_argument('--m_XY', dest='m_XY', type=int, default=16,
                        help='m_XY regularization threshold parameter')
    parser.add_argument('--learning_rate_exp', dest='learning_rate_exp',
                        type=int, default=2,
                        help='Learning rate negative exponent')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=.9,
                        help='Optimization momentum hyperparameter')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        default=128)
    X_params, Y_params = train(vars(parser.parse_args()))
    if not os.path.exists('../results'):
        os.path.makedirs('../results')
    hashing_utils.save_model(X_params, '../results/model_X.pkl')
    hashing_utils.save_model(Y_params, '../results/model_Y.pkl')
