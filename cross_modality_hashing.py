'''
Functions for mapping sequences to a common space
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
import hashing_utils
import collections


def train_cross_modality_hasher(X_train, X_train_mask, Y_train, Y_train_mask,
                                X_validate, X_validate_mask, Y_validate,
                                Y_validate_mask, hidden_layer_sizes,
                                alpha_XY, m_XY, output_dim, learning_rate=1e-5,
                                momentum=.9, batch_size=50, epoch_size=100,
                                initial_patience=1000,
                                improvement_threshold=0.99,
                                patience_increase=10, max_iter=100000):
    ''' Utility function for training a siamese net for cross-modality hashing
    So many parameters.
    Assumes X_train[n] should be mapped close to Y_train[m] only when n == m
    The number of hidden layers is inferred from the length of the entries of
    the hidden_layer_sizes dict.  A final dense output layer is also included.

    :parameters:
        - X_train, X_train_mask, Y_train, Y_train_mask, X_validate,
          X_validate_mask, Y_validate, Y_validate_mask : np.ndarray
            Training/validation sequences/masks in X/Y modality
            Sequence matrix shape=(n_sequences, n_time_steps, n_features)
            Mask matrix shape=(n_sequences, n_time_steps)
        - hidden_layer_sizes : dict of list-like
            Size of each hidden layer in X/Y network
        - alpha_XY : float
            Scaling parameter for cross-modality negative example cost
        - m_XY : int
            Cross-modality negative example threshold
        - output_dim : int
            Dimensonality of the output representation
        - learning_rate : float
            SGD learning rate
        - momentum : float
            SGD momentum
        - batch_size : int
            Mini-batch size
        - epoch_size : int
            Number of mini-batches per epoch
        - initial_patience : int
            Always train on at least this many batches
        - improvement_threshold : float
            Validation cost must decrease by this factor to increase patience
        - patience_increase : int
            How many more epochs should we wait when we increase patience
        - max_iter : int
            Maximum number of batches to train on

    :returns:
        - epoch : iterator
            Results for each epoch are yielded
    '''
    # Create networks
    layers = {
        'X': hashing_utils.build_network((None, None, X_train[0].shape[-1]),
                                         hidden_layer_sizes['X'], output_dim),
        'Y': hashing_utils.build_network((None, None, Y_train[0].shape[-1]),
                                         hidden_layer_sizes['Y'], output_dim)}
    # Inputs to X modality neural nets
    X_p_input = T.tensor3('X_p_input')
    X_p_mask = T.matrix('X_p_mask')
    X_n_input = T.tensor3('X_n_input')
    X_n_mask = T.matrix('X_n_mask')
    # Y network
    Y_p_input = T.tensor3('Y_p_input')
    Y_p_mask = T.matrix('Y_p_mask')
    Y_n_input = T.tensor3('Y_n_input')
    Y_n_mask = T.matrix('Y_n_mask')

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def hasher_cost(deterministic):
        X_p_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_p_input, layers['X']['mask']: X_p_mask},
            deterministic=deterministic)
        X_n_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_n_input, layers['X']['mask']: X_n_mask},
            deterministic=deterministic)
        Y_p_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_p_input, layers['Y']['mask']: Y_p_mask},
            deterministic=deterministic)
        Y_n_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_n_input, layers['Y']['mask']: Y_n_mask},
            deterministic=deterministic)
        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.mean(T.sum((X_p_output - Y_p_output)**2, axis=1))
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*hinge_cost(m_XY, X_n_output, Y_n_output)
        # Sum positive and negative costs for overall cost
        cost = cost_p + cost_n
        return cost

    # Combine all parameters from both networks
    params = (lasagne.layers.get_all_params(layers['X']['out'])
              + lasagne.layers.get_all_params(layers['Y']['out']))
    # Compute RMSProp gradient descent updates
    updates = lasagne.updates.rmsprop(hasher_cost(False), params,
                                      learning_rate, momentum)
    # Function for training the network
    train = theano.function([X_p_input, X_p_mask, X_n_input, X_n_mask,
                             Y_p_input, Y_p_mask, Y_n_input, Y_n_mask],
                            hasher_cost(False), updates=updates)

    # Compute cost without training
    cost = theano.function([X_p_input, X_p_mask, X_n_input, X_n_mask,
                            Y_p_input, Y_p_mask, Y_n_input, Y_n_mask],
                           hasher_cost(True))

    # Start with infinite validate cost; we will always increase patience once
    current_validate_cost = np.inf
    patience = initial_patience

    # Create fixed negative example validation set
    X_validate_shuffle = np.random.permutation(X_validate.shape[0])
    Y_validate_shuffle = X_validate_shuffle[
        hashing_utils.random_derangement(X_validate.shape[0])]
    data_iterator = hashing_utils.get_next_batch(
        X_train, X_train_mask, Y_train, Y_train_mask, batch_size, max_iter)
    # We will accumulate the mean train cost over each epoch
    train_cost = 0

    for n, (X_p, X_p_m, Y_p, Y_p_m,
            X_n, X_n_m, Y_n, Y_n_m) in enumerate(data_iterator):
        # Occasionally Theano was raising a MemoryError, this fails gracefully
        try:
            train_cost += train(X_p, X_p_m, X_n, X_n_m, Y_p, Y_p_m, Y_n, Y_n_m)
        except MemoryError:
            return
        # Stop training if a NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training cost {} at iteration {}'.format(train_cost, n)
            break
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Compute average training cost over the epoch
            epoch_result['train_cost'] = train_cost / float(epoch_size)
            # Reset training cost mean accumulation
            train_cost = 0
            # Also compute validate cost
            epoch_result['validate_cost'] = cost(
                X_validate, X_validate_mask,
                X_validate[X_validate_shuffle],
                X_validate_mask[X_validate_shuffle],
                Y_validate, Y_validate_mask,
                Y_validate[Y_validate_shuffle],
                Y_validate_mask[Y_validate_shuffle])

            if epoch_result['validate_cost'] < current_validate_cost:
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    patience += epoch_size*patience_increase
                current_validate_cost = epoch_result['validate_cost']

            # Yield scores and statistics for this epoch
            X_params = lasagne.layers.get_all_param_values(layers['X']['out'])
            Y_params = lasagne.layers.get_all_param_values(layers['Y']['out'])
            yield (epoch_result, X_params, Y_params)

            if n > patience:
                break

    return
