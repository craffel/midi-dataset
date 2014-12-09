'''
Functions for mapping data in different modalities to a common Hamming space
'''
import numpy as np
import theano.tensor as T
import theano
import nntools
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
                                alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y,
                                n_bits, dropout=True, learning_rate=1e-5,
                                momentum=.9, batch_size=100, epoch_size=1000,
                                initial_patience=10000,
                                improvement_threshold=0.99,
                                patience_increase=1.2, max_iter=200000,
                                mrr_samples=None):
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
        - alpha_X : float
            Scaling parameter for X-modality negative example cost
        - m_X : int
            Y-modality negative example threshold
        - alpha_Y : float
            Scaling parameter for Y-modality negative example cost
        - m_Y : int
            Y-modality negative example threshold
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
        - mrr_samples : int
            Indices of samples in the validation set over which to compute mean
            reciprocal rank.  None means use entire validation set.

    :returns:
        - epochs : list
            List of epoch dicts, which contains scores computed at each epoch
        - parameters : list
            List of NN parameters after each epoch
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
    layers_X.append(nntools.layers.InputLayer(
        shape=(batch_size, X_train.shape[1])))
    # Add each hidden layer recursively
    for num_units in hidden_layer_sizes_X:
        layers_X.append(nntools.layers.DenseLayer(
            layers_X[-1], num_units=num_units,
            nonlinearity=nntools.nonlinearities.tanh))
        if dropout:
            layers_X.append(nntools.layers.DropoutLayer(layers_X[-1]))
    # Add output layer
    layers_X.append(nntools.layers.DenseLayer(
        layers_X[-1], num_units=n_bits,
        nonlinearity=nntools.nonlinearities.tanh))

    # Y-modality hashing network
    layers_Y = []
    # As above
    layers_Y.append(nntools.layers.InputLayer(
        shape=(batch_size, Y_train.shape[1])))
    for num_units in hidden_layer_sizes_Y:
        layers_Y.append(nntools.layers.DenseLayer(
            layers_Y[-1], num_units=num_units,
            nonlinearity=nntools.nonlinearities.tanh))
        if dropout:
            layers_Y.append(nntools.layers.DropoutLayer(layers_Y[-1]))
    layers_Y.append(nntools.layers.DenseLayer(
        layers_Y[-1], num_units=n_bits,
        nonlinearity=nntools.nonlinearities.tanh))

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.sum((dist*(dist > 0))**2)

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
        cost_p = T.sum((X_p_output - Y_p_output)**2)
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*hinge_cost(m_XY, X_n_output, Y_n_output)
        # Thresholded, scaled cost of x-modality negative examples
        cost_x = alpha_X*hinge_cost(m_X, X_p_output, X_n_output)
        # Thresholded, scaled cost of y-modality negative examples
        cost_y = alpha_Y*hinge_cost(m_Y, Y_p_output, Y_n_output)
        # Return sum of these costs
        return cost_p + cost_n + cost_x + cost_y

    # Function for optimizing the neural net parameters, by minimizing cost
    params = (nntools.layers.get_all_params(layers_X[-1])
              + nntools.layers.get_all_params(layers_Y[-1]))
    updates = nntools.updates.nesterov_momentum(hasher_cost(False), params,
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
    X_output = layers_X[-1].get_output(X_input, deterministic=True)
    Y_output = layers_Y[-1].get_output(Y_input, deterministic=True)

    # A list of epoch result dicts, one per epoch
    epochs = []
    # A list of parameter settings at each epoch
    parameters = []

    # Create fixed negative example validation set
    X_validate_n = X_validate[np.random.permutation(X_validate.shape[0])]
    Y_validate_n = Y_validate[np.random.permutation(Y_validate.shape[0])]
    data_iterator = hashing_utils.get_next_batch(X_train, Y_train, batch_size,
                                                 max_iter)
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(data_iterator):
        train_cost = train(X_p, X_n, Y_p, Y_n)
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
            X_val_output = X_output.eval({X_input: X_validate})
            Y_val_output = Y_output.eval({Y_input: Y_validate})
            name = 'validate'
            N = X_val_output.shape[0]
            # Compute on the resulting hashes
            correct, in_mean, in_std = hashing_utils.statistics(
                X_val_output > 0, Y_val_output > 0)
            collisions, out_mean, out_std = hashing_utils.statistics(
                X_val_output[np.random.permutation(N)] > 0,
                Y_val_output > 0)
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

            if epoch_result['validate_cost'] < current_validate_cost:
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    patience *= patience_increase
                    print (" ... increasing patience to {} because "
                           "{} < {}*{}".format(patience,
                                               epoch_result['validate_cost'],
                                               improvement_threshold,
                                               current_validate_cost))
                current_validate_cost = epoch_result['validate_cost']
            # Only compute MRR on validate
            mrr_pessimist, mrr_optimist = hashing_utils.mean_reciprocal_rank(
                X_val_output[mrr_samples] > 0, Y_val_output > 0, mrr_samples)
            epoch_result['validate_mrr_pessimist'] = mrr_pessimist
            epoch_result['validate_mrr_optimist'] = mrr_optimist

            # Store scores and statistics for this epoch
            epochs.append(epoch_result)

            # Get current parameter settings
            current_parameters = collections.OrderedDict()
            for net_name, layers in zip(['X_net', 'Y_net'],
                                     [layers_X, layers_Y]):
                for n, layer in enumerate(layers):
                    for param in layer.get_params():
                        current_parameters["{}_layer_{}_{}".format(
                            net_name, n, param.name)] = param.get_value()
            # Store parameters for this epoch
            parameters.append(current_parameters)

            print '    patience : {}'.format(patience)
            print '    current_validation_cost : {}'.format(
                current_validate_cost)
            for k, v in epoch_result.items():
                print '    {} : {}'.format(k, round(v, 3))
            print

        if n > patience:
            break

    return epochs, parameters
