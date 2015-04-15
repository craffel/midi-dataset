'''
Functions for mapping data in different modalities to a common Hamming space
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
import hashing_utils
import collections


def train_cross_modality_hasher(X_train, Y_train, X_validate, Y_validate,
                                num_filters, filter_size, ds,
                                hidden_layer_sizes, alpha_XY, m_XY,
                                alpha_stress, n_bits=16, dropout=False,
                                learning_rate=.001, momentum=.0, batch_size=10,
                                sequence_length=100, epoch_size=100,
                                max_iter=50000, early_check=5000):
    ''' Utility function for training a siamese net for cross-modality hashing
    So many parameters.
    Assumes X_train[n] should be mapped close to Y_train[m] only when n == m
    The number of convolutional/pooling layers in inferred from the length of
    the entries in the num_filters, filter_size, ds dicts (all of which should
    have the same length).  The number of hidden layers is inferred from the
    length of the entries of the hidden_layer_sizes dict.  A final dense output
    layer is also included.

    :parameters:
        - X_train, Y_train, X_validate, Y_validate : list of np.ndarray
            List of train/validate sequences from X/Y modality
            Each shape=(n_channels, n_time_steps, n_features)
        - num_filters : dict of list-like
            Number of features in each convolutional layer for X/Y network
        - filter_size : dict of list-like
            Number of features in each convolutional layer for X/Y network
        - ds : dict of list-like
            Number of features in each convolutional layer for X/Y network
        - hidden_layer_sizes : dict of list-like
            Size of each hidden layer in X/Y network
        - alpha_XY : float
            Scaling parameter for cross-modality negative example cost
        - m_XY : int
            Cross-modality negative example threshold
        - alpha_stress : float
            Scaling hyperparameter for embedding stress cost
        - n_bits : int
            Number of bits in the output representation
        - dropout : bool
            Whether to use dropout between the hidden layers
        - learning_rate : float
            SGD learning rate
        - momentum : float
            SGD momentum
        - batch_size : int
            Mini-batch size
        - sequence_length : int
            Size of extracted sequences
        - epoch_size : int
            Number of mini-batches per epoch
        - max_iter : int
            Maximum number of iterations
        - early_check : int
            After this many iterations, if no progress has been made, quit

    :returns:
        - epoch : iterator
            Results for each epoch are yielded
    '''
    # First neural net, for X modality
    X_p_input = T.tensor4('X_p_input')
    X_n_input = T.tensor4('X_n_input')
    # For eval
    X_input = T.tensor4('X_input')
    # Second neural net, for Y modality
    Y_p_input = T.tensor4('Y_p_input')
    Y_n_input = T.tensor4('Y_n_input')
    Y_input = T.tensor4('Y_input')

    # Create networks
    layers = {
        'X': hashing_utils.build_network(
            (None, X_train[0].shape[0], sequence_length, X_train[0].shape[2]),
            num_filters['X'], filter_size['X'], ds['X'],
            hidden_layer_sizes['X'], dropout, n_bits),
        'Y': hashing_utils.build_network(
            (None, Y_train[0].shape[0], sequence_length, Y_train[0].shape[2]),
            num_filters['Y'], filter_size['Y'], ds['Y'],
            hidden_layer_sizes['Y'], dropout, n_bits)}

    # Compute pairwise cosine similarity of the rows of X
    def cosine_similarity(X):
        norms = T.sqrt(T.sum(X**2, axis=1))
        distance_matrix = T.dot(X, X.T)
        distance_matrix /= norms.reshape((-1, 1))
        distance_matrix /= norms.reshape((1, -1))
        return 1 - distance_matrix

    # Compute the stress between original and embedded distances
    def stress(original, embedded):
        tri_mask = 1 - np.tri(batch_size*sequence_length,
                            batch_size*sequence_length, 0,
                            dtype=theano.config.floatX)
        return T.sqrt(
            T.sum(tri_mask*(original - embedded)**2) /
            T.sum(tri_mask*original**2))

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def flatten_batch(X):
        return X.dimshuffle((0, 2, 1, 3)).reshape((-1, X.shape[1]*X.shape[3]))

    def hasher_cost(deterministic):
        X_p_output = layers['X'][-1].get_output(
            X_p_input, deterministic=deterministic)
        X_n_output = layers['X'][-1].get_output(
            X_n_input, deterministic=deterministic)
        Y_p_output = layers['Y'][-1].get_output(
            Y_p_input, deterministic=deterministic)
        Y_n_output = layers['Y'][-1].get_output(
            Y_n_input, deterministic=deterministic)

        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.mean((X_p_output - Y_p_output)**2)
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*hinge_cost(m_XY, X_n_output, Y_n_output)
        # Preserve distances with stress cost
        cost_stress = alpha_stress*(
            stress(cosine_similarity(flatten_batch(X_p_input)),
                   cosine_similarity(X_p_output))
            + stress(cosine_similarity(flatten_batch(Y_p_input)),
                     cosine_similarity(Y_p_output)))
        # Return sum of these costs
        return cost_p + cost_n + cost_stress

    # Combine all parameters from both networks
    params = (lasagne.layers.get_all_params(layers['X'][-1])
              + lasagne.layers.get_all_params(layers['Y'][-1]))
    # Compute RMSProp gradient descent updates
    updates = lasagne.updates.rmsprop(hasher_cost(False), params,
                                      learning_rate, momentum)
    # Function for training the network
    train = theano.function(
        [X_p_input, X_n_input, Y_p_input, Y_n_input], hasher_cost(False),
        updates=updates)

    # Functions for computing the neural net output on the train and val sets
    X_output = theano.function(
        [X_input], layers['X'][-1].get_output(X_input, deterministic=True))
    Y_output = theano.function(
        [Y_input], layers['Y'][-1].get_output(Y_input, deterministic=True))

    # Extract sample seqs from the validation set (only need to do this once)
    X_validate, Y_validate = hashing_utils.sample_sequences(
        X_validate, Y_validate, sequence_length)
    # Create fixed negative example validation set
    X_validate_shuffle = np.random.permutation(X_output(X_validate).shape[0])
    data_iterator = hashing_utils.get_next_batch(
        X_train, Y_train, batch_size, sequence_length, max_iter)
    # We will accumulate the mean train cost over each epoch
    train_cost = 0

    success = False

    for n, (X_p, Y_p, X_n, Y_n) in enumerate(data_iterator):
        # Occasionally Theano was raising a MemoryError, this fails gracefully
        try:
            train_cost += train(X_p, X_n, Y_p, Y_n)
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

            # Compute statistics on validation set
            X_val_output = X_output(X_validate)
            Y_val_output = Y_output(Y_validate)
            in_dist, in_mean, in_std = hashing_utils.statistics(
                X_val_output > 0, Y_val_output > 0)
            out_dist, out_mean, out_std = hashing_utils.statistics(
                X_val_output[X_validate_shuffle] > 0, Y_val_output > 0)
            epoch_result['validate_accuracy'] = in_dist[0]
            epoch_result['validate_in_class_distance_mean'] = in_mean
            epoch_result['validate_in_class_distance_std'] = in_std
            epoch_result['validate_collisions'] = out_dist[0]
            epoch_result['validate_out_of_class_distance_mean'] = out_mean
            epoch_result['validate_out_of_class_distance_std'] = out_std
            X_entropy = hashing_utils.hash_entropy(X_val_output > 0)
            epoch_result['validate_hash_entropy_X'] = X_entropy
            Y_entropy = hashing_utils.hash_entropy(Y_val_output > 0)
            epoch_result['validate_hash_entropy_Y'] = Y_entropy
            # Objective is the ratio of accurate hashes to collisions
            # We should try to maximize it
            # When either is small, it's not really valid
            if out_dist[0] > 1e-5 and in_dist[0] > 1e-2:
                bhatt_coeff = -np.sum(np.sqrt(in_dist*out_dist))
                epoch_result['validate_objective'] = bhatt_coeff
                success = True
            else:
                epoch_result['validate_objective'] = -1

            # If we haven't had a successful epoch yet, quit early
            if n >= early_check and not success:
                return

            # Yield scores and statistics for this epoch
            X_params = lasagne.layers.get_all_param_values(layers['X'][-1])
            Y_params = lasagne.layers.get_all_param_values(layers['Y'][-1])
            yield (epoch_result, X_params, Y_params)

    return
