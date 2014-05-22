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
improvement_threshold = 0.99
# Amount to increase patience when validation cost has decreased
patience_increase = 1.2
# Maximum number of batches to train on
max_iter = 200*epoch_size
# Use this many samples to compute mean reciprocal rank
n_mrr_samples = 500

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

# Pre-compute indices over which to compute mrr
mrr_samples = np.random.choice(X_validate.shape[1], n_mrr_samples, False)

# Create fixed negative example validation set
X_validate_n = X_validate[:, np.random.permutation(X_validate.shape[1])]
Y_validate_n = Y_validate[:, np.random.permutation(Y_validate.shape[1])]

while True:
    # Randomly choose a value for each hyperparameter
    hp = parameter_space_3()
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
    
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(hashing_utils.get_next_batch(X_train, Y_train, batch_size, max_iter)):
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
                correct, in_class_mean, in_class_std = hashing_utils.statistics(X_output > 0, Y_output > 0)
                collisions, out_of_class_mean, out_of_class_std = hashing_utils.statistics(X_output[:, np.random.permutation(N)] > 0,
                                                                                           Y_output > 0)
                epoch_result[name + '_accuracy'] = correct/float(N)
                epoch_result[name + '_in_class_distance_mean'] = in_class_mean
                epoch_result[name + '_in_class_distance_std'] = in_class_std
                epoch_result[name + '_collisions'] = collisions/float(N)
                epoch_result[name + '_out_of_class_distance_mean'] = out_of_class_mean
                epoch_result[name + '_out_of_class_distance_std'] = out_of_class_std
                epoch_result[name + '_hash_entropy_X'] = hashing_utils.hash_entropy(X_output > 0)
                epoch_result[name + '_hash_entropy_Y'] = hashing_utils.hash_entropy(Y_output > 0)
            if epoch_result['validate_cost'] < current_validate_cost:
                if epoch_result['validate_cost'] < improvement_threshold*current_validate_cost:
                    patience *= patience_increase
                    print " ... increasing patience to {} because {} < {}*{}".format(patience,
                                                                                     epoch_result['validate_cost'],
                                                                                     improvement_threshold,
                                                                                     current_validate_cost)
                current_validate_cost = epoch_result['validate_cost']
            # Only compute MRR on validate
            mrr_pessimist, mrr_optimist = hashing_utils.mean_reciprocal_rank(X_output[:, mrr_samples] > 0,
                                                                             Y_output > 0,
                                                                             mrr_samples)
            epoch_result['validate_mrr_pessimist'] = mrr_pessimist
            epoch_result['validate_mrr_optimist'] = mrr_optimist

            epoch_results.append(epoch_result)
            print '    patience : {}'.format(patience)
            print '    current_validation_cost : {}'.format(current_validate_cost)
            for k, v in epoch_result.items():
                print '    {} : {}'.format(k, round(v, 3))
            print
        if n > patience:
            break
    with open(os.path.join(trial_directory, 'epochs.pkl'), 'wb') as f:
        pickle.dump(epoch_results, f)

