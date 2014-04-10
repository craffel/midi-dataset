# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('../')
import cross_modality_hashing
import numpy as np
import theano.tensor as T
import theano
import glob
import matplotlib.pyplot as plt
from IPython import display
import os
import pickle
import pprint
import collections
import shutil
import scipy.spatial
import scipy.weave

# <codecell>

def shingle(x, stacks):
    ''' Shingle a matrix column-wise '''
    return np.vstack([x[:, n:(x.shape[1] - stacks + n)] for n in xrange(stacks)])

# <codecell>

def load_data(directory, shingle_size=4, train_validate_split=.9):
    ''' Load in all chroma matrices and piano rolls and output them as separate matrices '''
    X_train = []
    Y_train = []
    X_validate = []
    Y_validate = []
    for chroma_filename in glob.glob(os.path.join(directory, '*-msd.npy')):
        piano_roll_filename = chroma_filename.replace('msd', 'midi')
        if np.random.rand() < train_validate_split:
            X_train.append(shingle(np.load(chroma_filename), shingle_size))
            Y_train.append(shingle(np.load(piano_roll_filename), shingle_size))
        else:
            X_validate.append(shingle(np.load(chroma_filename), shingle_size))
            Y_validate.append(shingle(np.load(piano_roll_filename), shingle_size))
    return np.array(np.hstack(X_train), dtype=theano.config.floatX, order='F'), \
           np.array(np.hstack(Y_train), dtype=theano.config.floatX, order='F'), \
           np.array(np.hstack(X_validate), dtype=theano.config.floatX, order='F'), \
           np.array(np.hstack(Y_validate), dtype=theano.config.floatX, order='F')

# <codecell>

def get_next_batch(X, Y, batch_size, n_iter):
    ''' Fast (hopefully) random mini batch generator '''
    N = X.shape[1]
    n_batches = int(np.floor(N/float(batch_size)))
    current_batch = n_batches
    for n in xrange(n_iter):
        if current_batch >= n_batches:
            positive_shuffle = np.random.permutation(N)
            negative_shuffle = np.random.permutation(N)
            X_p = np.array(X[:, positive_shuffle])
            Y_p = np.array(Y[:, positive_shuffle])
            #X_n = np.array(X[:, np.mod(negative_shuffle + 2*np.random.randint(0, 2, N) - 1, N)])
            X_n = np.array(X[:, np.random.permutation(N)])
            Y_n = np.array(Y[:, negative_shuffle])
            current_batch = 0
        batch = slice(current_batch*batch_size, (current_batch + 1)*batch_size)
        yield X_p[:, batch], Y_p[:, batch], X_n[:, batch], Y_n[:, batch]
        current_batch += 1

# <codecell>

def standardize(X):
    ''' Return column vectors to standardize X, via (X - X_mean)/X_std '''
    std = np.std(X, axis=1).reshape(-1, 1)
    return np.mean(X, axis=1).reshape(-1, 1), std + (std == 0)

# <codecell>

def hash_entropy(X):
        ''' Get the entropy of the histogram of hashes (want this to be close to n_bits) '''
        bit_values = np.sum(2**np.arange(X.shape[0]).reshape(-1, 1)*X, axis=0)
        counts, _ = np.histogram(bit_values, np.arange(2**X.shape[0]))
        counts = counts/float(counts.sum())
        return -np.sum(counts*np.log2(counts + 1e-100))

# <codecell>

def statistics(X, Y):
    ''' Computes the number of correctly encoded codeworks and the number of bit errors made '''
    points_equal = (X == Y)
    return np.all(points_equal, axis=0).sum(), \
           np.mean(np.logical_not(points_equal).sum(axis=0)), \
           np.std(np.logical_not(points_equal).sum(axis=0))

# <codecell>

def fast_binary_distance(X, Y):
    '''
    Compute the binary (matching) distance between all columns of X and all columns of Y
    
    Input:
        X - M x N binary matrix
        Y - M x K binary matrix
    Output:
        D - N x K distance matrix such that D[i, j] = sum(X[:, i] != Y[:, j])
    '''
    
    (M, N) = X.shape
    assert Y.shape[0] == M
    (M, K) = Y.shape
    assert X.dtype == np.bool
    assert Y.dtype == np.bool

    D = np.zeros((N, K), dtype=np.int)
    
    weaver = r"""
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                D[i*N + j] += X[k*N + i]^Y[k*N + j];
            }
        }
    }
"""
    scipy.weave.inline(weaver, arg_names=['M', 'N', 'K', 'X', 'Y', 'D'])
    return D

# <codecell>

def mean_reciprocal_rank(X, Y):
    ''' Computes the mean reciprocal rank of the correct codeword '''
    # Compute distances between each codeword and each other codeword
    distance_matrix = fast_binary_distance(X, Y)
    # Rank is the number of distances smaller than the correct distance, which is the value on the diagonal
    return np.mean(1./(distance_matrix.T <= np.diag(distance_matrix)).sum(axis=0))

# <codecell>

# First neural net, for chroma vectors
X_p_input = T.matrix('X_p_input')
X_n_input = T.matrix('X_n_input')
# Second neural net, for MIDI piano roll
Y_p_input = T.matrix('Y_p_input')
Y_n_input = T.matrix('Y_n_input')
# Symbolic hyperparameters
alpha_XY = T.scalar('alpha_XY')
m_XY = T.scalar('m_XY')
alpha_X = T.scalar('alpha_X')
m_X = T.scalar('m_X')
alpha_Y = T.scalar('alpha_Y')
m_Y = T.scalar('m_Y')
# SGD learning rate
learning_rate = 1e-4
# SGD momentum
momentum = .9
# Mini-batch size
batch_size = 10
# Number of mini-batches per epoch
epoch_size = 1000
# Always train on at least this many batches
initial_patience = 10000
# Validation cost must decrease by this factor to increase patience
improvement_threshold = 0.98
# Amount to increase patience when validation cost has decreased
patience_increase = 1.2
# Maximum number of batches to train on
max_iter = int(1e8)
# Use this many samples to compute mean reciprocal rank
n_mrr_samples = 1000

# Possible values for each hyperparameter to take
hp_values = collections.OrderedDict()
hp_values['n_bits'] = [8, 12, 16]
hp_values['n_layers'] = [3, 4]
hp_values['alpha_XY'] = np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX)
hp_values['m_XY'] = np.arange(17)
hp_values['alpha_X'] = np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX)
hp_values['m_X'] = np.arange(17)
hp_values['alpha_Y'] = np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX)
hp_values['m_Y'] = np.arange(17)
# Current value of hyperparameters for one trial
hp = collections.OrderedDict()

# Set up paths
base_data_directory = '../data'
result_directory = os.path.join(base_data_directory, 'parameter_search')
training_data_directory = os.path.join(base_data_directory, 'hash_dataset')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
# Load in the data
X_train, Y_train, X_validate, Y_validate = load_data(training_data_directory)

# Standardize
X_mean, X_std = standardize(X_train)
X_train = (X_train - X_mean)/X_std
X_validate = (X_validate - X_mean)/X_std
Y_mean, Y_std = standardize(Y_train)
Y_train = (Y_train - Y_mean)/Y_std
Y_validate = (Y_validate - Y_mean)/Y_std
# Create fixed negative example validation set
X_validate_n = X_validate[:, np.random.permutation(X_validate.shape[1])]
Y_validate_n = Y_validate[:, np.random.permutation(Y_validate.shape[1])]

while True:
    # Randomly choose a value for each hyperparameter
    for hyperparameter, values in hp_values.items():
        hp[hyperparameter] = np.random.choice(values)
    # A list of results dicts, one per epoch
    epoch_results = []
    # Make a subdirectory for this parameter setting
    parameter_string = ','.join(["{}={}".format(k, round(v, 2)) for (k, v) in hp.items()])
    print
    print
    print "##################"
    print parameter_string
    print "##################"
    trial_directory = os.path.join(result_directory, parameter_string)
    # In the very odd case that we have already tried this parameter setting, skip
    if os.path.exists(trial_directory):
        continue
    os.makedirs(trial_directory)
    # Save the parameter dict
    with open(os.path.join(trial_directory, 'parameters.pkl'), 'wb') as f:
        pickle.dump(hp, f)
    
    # Compute layer sizes.  Middle layers are nextpow2(input size)
    hidden_layer_size_X = int(2**np.ceil(np.log2(X_train.shape[0])))
    layer_sizes_x = [X_train.shape[0]] + [hidden_layer_size_X]*(hp['n_layers'] - 1) + [hp['n_bits']]
    hidden_layer_size_Y = int(2**np.ceil(np.log2(X_train.shape[0])))
    layer_sizes_y = [Y_train.shape[0]] + [hidden_layer_size_Y]*(hp['n_layers'] - 1) + [hp['n_bits']]
    hasher = cross_modality_hashing.SiameseNet(layer_sizes_x, layer_sizes_y)
    
    # Create theano symbolic function for cost
    hasher_cost = hasher.cross_modality_cost(X_p_input, X_n_input, Y_p_input, Y_n_input,
                                             alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y)
    # Function for optimizing the neural net parameters, by minimizing cost
    train = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input,
                             alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y],
                            hasher_cost,
                            updates=cross_modality_hashing.gradient_updates_momentum(hasher_cost,
                                                                                     hasher.params,
                                                                                     learning_rate,
                                                                                     momentum))
    # Compute cost without trianing
    cost = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input,
                             alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y], hasher_cost)

    # Keep track of the patience - we will always increase the patience once
    patience = initial_patience/patience_increase 
    current_validate_cost = np.inf
    
    # Functions for computing the neural net output on the train and validation sets
    X_train_output = hasher.X_net.output(X_train)
    Y_train_output = hasher.Y_net.output(Y_train)
    X_validate_output = hasher.X_net.output(X_validate)
    Y_validate_output = hasher.Y_net.output(Y_validate)
    
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(get_next_batch(X_train, Y_train, batch_size, max_iter)):
        train_cost = train(X_p, X_n, Y_p, Y_n, hp['alpha_XY'], hp['m_XY'], hp['alpha_X'], hp['m_X'], hp['alpha_Y'], hp['m_Y'])
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Store current SGD cost
            epoch_result['train_cost'] = train_cost
            # Also compute validate cost (more stable)
            epoch_result['validate_cost'] = cost(X_validate, X_validate_n, Y_validate, Y_validate_n, hp['alpha_XY'],
                                                 hp['m_XY'], hp['alpha_X'], hp['m_X'], hp['alpha_Y'], hp['m_Y'])
            
            # Get accuracy and diagnostic figures for both train and validation sets
            for name, X_output, Y_output in [('train', X_train_output.eval(), Y_train_output.eval()),
                                             ('validate', X_validate_output.eval(), Y_validate_output.eval())]:
                N = X_output.shape[1]
                # Compute and display metrics on the resulting hashes
                correct, in_class_mean, in_class_std = statistics(X_output > 0, Y_output > 0)
                collisions, out_of_class_mean, out_of_class_std = statistics(X_output[:, np.random.permutation(N)] > 0,
                                                                             Y_output > 0)
                epoch_result[name + '_accuracy'] = correct/float(N)
                epoch_result[name + '_in_class_distance_mean'] = in_class_mean
                epoch_result[name + '_in_class_distance_std'] = in_class_std
                epoch_result[name + '_collisions'] = collisions/float(N)
                epoch_result[name + '_out_of_class_distance_mean'] = out_of_class_mean
                epoch_result[name + '_out_of_class_distance_std'] = out_of_class_std
                epoch_result[name + '_hash_entropy_X'] = hash_entropy(X_output > 0)
                epoch_result[name + '_hash_entropy_Y'] = hash_entropy(Y_output > 0)
                mrr_samples = np.random.choice(N, n_mrr_samples, False)
                epoch_result[name + '_mean_reciprocal_rank'] = mean_reciprocal_rank(X_output[:, mrr_samples] > 0,
                                                                                    Y_output[:, mrr_samples] > 0)
            
            if epoch_result['validate_cost'] < improvement_threshold*current_validate_cost:
                patience *= patience_increase
                print " ... increasing patience to {} because {} < {}*{}".format(patience,
                                                                                 epoch_result['validate_cost'],
                                                                                 improvement_threshold,
                                                                                 current_validate_cost)
                current_validate_cost = epoch_result['validate_cost']
                
            epoch_results.append(epoch_result)
            for k, v in epoch_result.items():
                print '    {} : {}'.format(k, round(v, 3))
            print
        if n > patience:
            break
    with open(os.path.join(trial_directory, 'epochs.pkl'), 'wb') as f:
        pickle.dump(epoch_results, f)

