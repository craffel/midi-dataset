'''
Functions for mapping data in different modalities to a common Hamming space
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
import hashing_utils
import collections


# Cost is given by
# $$
# \mathcal{L}_{XY} = \frac{1}{2} \sum_{(x, y) \in \mathcal{P}_{XY}} \|\xi(x) - \eta(y)\|_2^2
# + \frac{1}{2} \sum_{(x, y) \in \mathcal{N}_{XY}} \max \{0, m_{XY} - \|\xi(x) - \eta(y)\|_2\}^2
# $$
# where $(x, y) \in \mathcal{P}_{XY}$ denotes that $x$ and $y$ are in the same
# class and $(x, y) \in \mathcal{N}_{XY}$ denotes that $x$ and $y$ are in
# different classes, $\xi$ is the function of one neural network and $\eta$ is
# the function of the other and $m_{XY}$ is the hinge loss threshold.


def train_cross_modality_hasher(X_train, Y_train, X_validate, Y_validate,
                                hidden_layer_sizes_X, hidden_layer_sizes_Y,
                                alpha_XY, m_XY, n_bits, dropout=False,
                                learning_rate=1e-5, momentum=.9,
                                batch_size=100, epoch_size=1000,
                                initial_patience=10000,
                                improvement_threshold=0.995,
                                patience_increase=1.1, max_iter=200000):
    ''' Utility function for training a siamese net for cross-modality hashing
    So many parameters.
    Assumes X_train[n] should be mapped close to Y_train[m] only when n == m

    :parameters:
        - X_train : np.ndarray, shape=(n_train_examples, n_features)
            Training data in X modality
        - Y_train : np.ndarray, shape=(n_train_example, n_features)
            Training data in Y modality
        - X_validate : np.ndarray, shape=(n_validate_examples, n_features)
            Validation data in X modality
        - Y_validate : np.ndarray, shape=(n_validate_examples, n_features)
            Validation data in Y modality
        - hidden_layer_sizes_X : list-like
            Size of each hidden layer in X network.
            Number of layers = len(hidden_layer_sizes_X) + 1
        - hidden_layer_sizes_Y : list-like
            Size of each hidden layer in Y network.
            Number of layers = len(hidden_layer_sizes_Y) + 1
        - alpha_XY : float
            Scaling parameter for cross-modality negative example cost
        - m_XY : int
            Cross-modality negative example threshold
        - n_bits : int
            Number of bits in the output representation
        - dropout : bool
            Whether to use dropout
        - learning_rate : float
            SGD learning rate
        - momemntum : float
            SGD momentum
        - batch_size : int
            Mini-batch size
        - epoch_size : int
            Number of mini-batches per epoch
        - initial_patience : int
            Always train on at least this many batches
        - improvement_threshold : float
            Validation cost must decrease by this factor to increase patience
        - patience_increase : float
            Amount to increase patience when validation cost has decreased
        - max_iter : int
            Maximum number of batches to train on, default 200000

    :returns:
        - epoch : iterator
            Results for each epoch are yielded
    '''
    # First neural net, for X modality
    X_p_input = T.matrix('X_p_input')
    X_n_input = T.matrix('X_n_input')
    # For eval
    X_input = T.matrix('X_input')
    # Second neural net, for Y modality
    Y_p_input = T.matrix('Y_p_input')
    Y_n_input = T.matrix('Y_n_input')
    Y_input = T.matrix('Y_input')

    # X-modality hashing network
    layers_X = []
    # Start with input layer
    layers_X.append(lasagne.layers.InputLayer(
        shape=(batch_size, X_train.shape[1])))
    # Add each hidden layer recursively
    for num_units in hidden_layer_sizes_X:
        layers_X.append(lasagne.layers.DenseLayer(
            layers_X[-1], num_units=num_units,
            nonlinearity=lasagne.nonlinearities.tanh))
        if dropout:
            layers_X.append(lasagne.layers.DropoutLayer(layers_X[-1]))
    # Add output layer
    layers_X.append(lasagne.layers.DenseLayer(
        layers_X[-1], num_units=n_bits,
        nonlinearity=lasagne.nonlinearities.tanh))

    # Y-modality hashing network
    layers_Y = []
    # As above
    layers_Y.append(lasagne.layers.InputLayer(
        shape=(batch_size, Y_train.shape[1])))
    for num_units in hidden_layer_sizes_Y:
        layers_Y.append(lasagne.layers.DenseLayer(
            layers_Y[-1], num_units=num_units,
            nonlinearity=lasagne.nonlinearities.tanh))
        if dropout:
            layers_Y.append(lasagne.layers.DropoutLayer(layers_Y[-1]))
    layers_Y.append(lasagne.layers.DenseLayer(
        layers_Y[-1], num_units=n_bits,
        nonlinearity=lasagne.nonlinearities.tanh))

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def hasher_cost(deterministic):
        X_p_output = layers_X[-1].get_output(X_p_input,
                                             deterministic=deterministic)
        X_n_output = layers_X[-1].get_output(X_n_input,
                                             deterministic=deterministic)
        Y_p_output = layers_Y[-1].get_output(Y_p_input,
                                             deterministic=deterministic)
        Y_n_output = layers_Y[-1].get_output(Y_n_input,
                                             deterministic=deterministic)

        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.mean((X_p_output - Y_p_output)**2)
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*hinge_cost(m_XY, X_n_output, Y_n_output)
        # Return sum of these costs
        return cost_p + cost_n

    # Function for optimizing the neural net parameters, by minimizing cost
    params = (lasagne.layers.get_all_params(layers_X[-1])
              + lasagne.layers.get_all_params(layers_Y[-1]))
    updates = lasagne.updates.nesterov_momentum(hasher_cost(False), params,
                                                learning_rate, momentum)
    train = theano.function(
        [X_p_input, X_n_input, Y_p_input, Y_n_input], hasher_cost(False),
        updates=updates)
    # Compute cost without trianing
    cost = theano.function(
        [X_p_input, X_n_input, Y_p_input, Y_n_input], hasher_cost(True))

    # Keep track of the patience - we will always increase the patience once
    patience = initial_patience/patience_increase
    current_validate_cost = np.inf

    # Functions for computing the neural net output on the train and val sets
    X_output = theano.function(
        [X_input], layers_X[-1].get_output(X_input, deterministic=True))
    Y_output = theano.function(
        [Y_input], layers_Y[-1].get_output(Y_input, deterministic=True))

    # Create fixed negative example validation set
    X_validate_n = X_validate[np.random.permutation(X_validate.shape[0])]
    Y_validate_n = Y_validate[np.random.permutation(Y_validate.shape[0])]
    X_validate_shuffle = np.random.permutation(X_validate.shape[0])
    data_iterator = hashing_utils.get_next_batch(X_train, Y_train, batch_size,
                                                 max_iter)
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(data_iterator):
        train_cost = train(X_p, X_n, Y_p, Y_n)
        if not np.isfinite(train_cost):
            print 'Bad training cost {} at iteration {}'.format(train_cost, n)
            break
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Store current SGD cost
            epoch_result['train_cost'] = train_cost
            # Also compute validate cost (more stable)
            epoch_result['validate_cost'] = cost(
                X_validate, X_validate_n, Y_validate, Y_validate_n)

            # Compute statistics on validation set only
            X_val_output = X_output(X_validate)
            Y_val_output = Y_output(Y_validate)
            name = 'validate'
            # Compute on the resulting hashes
            correct, in_mean, in_std = hashing_utils.statistics(
                X_val_output > 0, Y_val_output > 0)
            collisions, out_mean, out_std = hashing_utils.statistics(
                X_val_output[X_validate_shuffle] > 0,
                Y_val_output > 0)
            N = X_val_output.shape[0]
            epoch_result[name + '_accuracy'] = correct/float(N)
            epoch_result[name + '_in_class_distance_mean'] = in_mean
            epoch_result[name + '_in_class_distance_std'] = in_std
            epoch_result[name + '_collisions'] = collisions/float(N)
            epoch_result[name + '_out_of_class_distance_mean'] = out_mean
            epoch_result[name + '_out_of_class_distance_std'] = out_std
            epoch_result[name + '_hash_entropy_X'] = \
                hashing_utils.hash_entropy(X_val_output > 0)
            epoch_result[name + '_hash_entropy_Y'] = \
                hashing_utils.hash_entropy(Y_val_output > 0)
            epoch_result[name + '_objective'] = in_mean/out_mean

            if epoch_result['validate_cost'] < current_validate_cost:
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    patience *= patience_increase
                current_validate_cost = epoch_result['validate_cost']

            # Store scores and statistics for this epoch
            yield epoch_result, X_output, Y_output

        if n > patience:
            break

    return
