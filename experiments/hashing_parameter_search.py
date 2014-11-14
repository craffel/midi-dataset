# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import theano.tensor as T
import theano
import os
import pickle
import collections

# <codecell>

def parameter_space_1():
    ''' Randomly sample the first parameter search space '''
    hp_values = collections.OrderedDict()
    hp_values['n_bits'] = np.random.choice([8, 12, 16])
    hp_values['n_layers'] = np.random.choice([3, 4])
    hp_values['alpha_XY'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_XY'] = np.random.choice(17)
    hp_values['alpha_X'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_X'] = np.random.choice(17)
    hp_values['alpha_Y'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_Y'] = np.random.choice(17)
    return hp_values

# <codecell>

def parameter_space_2():
    # Possible values for each hyperparameter to take
    hp_values = collections.OrderedDict()
    hp_values['n_bits'] = np.random.choice([16, 24])
    hp_values['n_layers'] = 3
    hp_values['alpha_XY'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_XY'] = np.random.choice(17)
    hp_values['alpha_X'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_X'] = np.random.choice(17)
    hp_values['alpha_Y'] = np.random.choice(np.array(np.linspace(0, 2, 201), dtype=theano.config.floatX))
    hp_values['m_Y'] = np.random.choice(17)
    return hp_values

# <codecell>

def parameter_space_3():
    hp_values = collections.OrderedDict()
    # 24 bits works better than 16
    hp_values['n_bits'] = 24
    hp_values['n_layers'] = 3
    # Total alpha should be between 3 and 6
    total_alpha = 3*np.random.random_sample() + 3
    # Compute random proportion of each alpha
    alpha_proportions = np.random.random_sample(3)
    alpha_proportions /= alpha_proportions.sum()
    hp_values['alpha_XY'] = np.array([total_alpha*alpha_proportions[0]], dtype=theano.config.floatX)[0]
    hp_values['m_XY'] = np.random.choice(10)
    hp_values['alpha_X'] = np.array([total_alpha*alpha_proportions[1]], dtype=theano.config.floatX)[0]
    hp_values['m_X'] = np.random.choice(10)
    hp_values['alpha_Y'] = np.array([total_alpha*alpha_proportions[2]], dtype=theano.config.floatX)[0]
    hp_values['m_Y'] = np.random.choice(10)
    return hp_values

# <codecell>

# Set up paths
base_data_directory = '../data'
result_directory = os.path.join(base_data_directory, 'parameter_search_3')
training_data_directory = os.path.join(base_data_directory, 'hash_dataset')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
# Load in the data
X, Y = hashing_utils.load_data(training_data_directory)
# Split into train and validate and standardize
X_train, Y_train, X_validate, Y_validate = hashing_utils.train_validate_split(X, Y)

# Use this many samples to compute mean reciprocal rank
n_mrr_samples = 500
# Pre-compute indices over which to compute mrr
mrr_samples = np.random.choice(X_validate.shape[1], n_mrr_samples, False)

# Compute layer sizes.  Middle layers are nextpow2(input size)
hidden_layer_size_X = int(2**np.ceil(np.log2(X_train.shape[0])))
hidden_layer_size_Y = int(2**np.ceil(np.log2(Y_train.shape[0])))

while True:
    # Randomly choose a value for each hyperparameter
    hp = parameter_space_3()
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
    # Save the hyperparameter dict
    with open(os.path.join(trial_directory, 'hyperparameters.pkl'), 'wb') as f:
        pickle.dump(hp, f)
    
    epochs, parameters = cross_modality_hashing.train_cross_modality_hasher(X_train, Y_train, X_validate, Y_validate, 
                             [hidden_layer_size_X]*(hp['n_layers'] - 1),
                             [hidden_layer_size_Y]*(hp['n_layers'] - 1),
                             hp['alpha_XY'], hp['m_XY'], hp['alpha_X'], hp['m_X'], hp['alpha_Y'],
                             hp['m_Y'], hp['n_bits'], mrr_samples=mrr_samples)

    with open(os.path.join(trial_directory, 'epochs.pkl'), 'wb') as f:
        pickle.dump(epoch_results, f)
    with open(os.path.join(trial_directory, 'parameters.pkl'), 'wb') as f:
        pickle.dump(epoch_results, f)

