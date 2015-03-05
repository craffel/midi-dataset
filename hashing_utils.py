'''
Utilities for cross-modality hashing experiments, including data loading and
statistics
'''
import numpy as np
import theano
import scipy.spatial
import pickle
import lasagne
import collections


def shingle(x, stacks):
    ''' Shingles a matrix column-wise

    :parameters:
        - x : np.ndarray
            Matrix to shingle
        - stacks : int
            Number of copies of each column to stack

    :returns:
        - x_shingled : np.ndarray
            X with columns stacked
    '''
    return np.hstack([x[n:(x.shape[0] - stacks + n)]
                      for n in xrange(stacks)])


def standardize(X):
    ''' Return column vectors to standardize X, via (X - X_mean)/X_std

    :parameters:
        - X : np.ndarray, shape=(n_examples, n_features)
            Data matrix

    :returns:
        - X_mean : np.ndarray, shape=(n_features, 1)
            Mean column vector
        - X_std : np.ndarray, shape=(n_features, 1)
            Standard deviation column vector
    '''
    std = np.std(X, axis=0)
    return np.mean(X, axis=0), std + (std == 0)


def load_data(train_list, valid_list, shingle_size=4):
    '''
    Load in dataset given lists of files in each split.
    Also shindles and standardizes (using train mean/std) the data.
    Each file should be a .npz file with a key 'X' for data in the X modality
    and 'Y' for data in the Y modality.

    :parameters:
        - train_list : list of str
            List of paths to files in the training set.
        - valid_list : list of str
            List of paths to files in the validation set.
        - shingle_size : int
            Number of entries to shingle.

    :returns:
        - X_train : list
            List of np.ndarrays of X modality features in training set
        - Y_train : list
            List of np.ndarrays of Y modality features in training set
        - X_valid : list
            List of np.ndarrays of X modality features in validation set
        - Y_valid : list
            List of np.ndarrays of Y modality features in validation set
    '''
    # We'll use dicts where key is the data subset, so we can iterate
    X = collections.defaultdict(list)
    Y = collections.defaultdict(list)
    for file_list, key in zip([train_list, valid_list],
                              ['train', 'valid']):
        # Load in all files
        for filename in file_list:
            data = np.load(filename)
            # Shingle and convert to floatX with correct column order
            X[key].append(np.array(shingle(data['X'], shingle_size),
                                   dtype=theano.config.floatX, order='C'))
            Y[key].append(np.array(shingle(data['Y'], shingle_size),
                                   dtype=theano.config.floatX, order='C'))
        # Stack all examples into big matrix
        X[key] = np.vstack(X[key])
        Y[key] = np.vstack(Y[key])
        # Get mean/std for training set
        if key == 'train':
            X_mean, X_std = standardize(X[key])
            Y_mean, Y_std = standardize(Y[key])
        # Use training set mean/std to standardize
        X[key] = (X[key] - X_mean)/X_std
        Y[key] = (Y[key] - Y_mean)/Y_std

    return X['train'], Y['train'], X['valid'], Y['valid']


def get_next_batch(X, Y, batch_size, n_iter):
    ''' Randomly generates positive and negative example minibatches

    :parameters:
        - X : np.ndarray, shape=(n_examples, n_features)
            Data matrix in one modality
        - y : np.ndarray, shape=(n_examples, n_features)
            Data matrix in another modality
        - batch_size : int
            Size of each minibatch to grab
        - n_iter : int
            Total number of iterations to run

    :returns:
        - X_p : np.ndarray
            Positive example minibatch in X modality
        - Y_p : np.ndarray
            Positive example minibatch in Y modality
        - X_n : np.ndarray
            Negative example minibatch in X modality
        - Y_n : np.ndarray
            Negative example minibatch in Y modality
    '''
    N = X.shape[0]
    n_batches = int(np.floor(N/float(batch_size)))
    current_batch = n_batches
    for n in xrange(n_iter):
        if current_batch >= n_batches:
            positive_shuffle = np.random.permutation(N)
            negative_shuffle = np.random.permutation(N)
            X_p = np.array(X[positive_shuffle])
            Y_p = np.array(Y[positive_shuffle])
            # X_n = np.array(X[:, np.mod(negative_shuffle +
            #                            2*np.random.randint(0, 2, N) - 1, N)])
            X_n = np.array(X[np.random.permutation(N)])
            Y_n = np.array(Y[negative_shuffle])
            current_batch = 0
        batch = slice(current_batch*batch_size, (current_batch + 1)*batch_size)
        yield X_p[batch], Y_p[batch], X_n[batch], Y_n[batch]
        current_batch += 1


def hash_entropy(X):
    ''' Get the entropy of the histogram of hashes.
    We want this to be close to n_bits.

    :parameters:
        - X : np.ndarray, shape=(n_examples, n_bits)
            Boolean data matrix, each column is the hash of an example

    :returns:
        - hash_entropy : float
            Entropy of the hash distribution
    '''
    # Convert bit vectors to ints
    bit_values = np.sum(2**np.arange(X.shape[1])*X, axis=1)
    # Count the number of occurences of each int
    counts = np.bincount(bit_values)
    # Normalize to form a probability distribution
    counts = counts/float(counts.sum())
    # Compute entropy
    return -np.sum(counts*np.log2(counts + 1e-100))


def statistics(X, Y):
    ''' Computes the number of correctly encoded codeworks and the number of
    bit errors made.  Assumes that rows of X should be hashed the same as rows
    of Y

    :parameters:
        - X : np.ndarray, shape=(n_examples, n_features)
            Data matrix of X modality
        - Y : np.ndarray, shape=(n_examples, n_features)
            Codeword matrix of Y modality

    :returns:
        - distance_distribution : int
            Emprical distribution of the codeword distances
        - mean_distance : float
            Mean of distances between corresponding codewords
        - std_distance : float
            Std of distances between corresponding codewords
    '''
    points_equal = (X == Y)
    distances = np.logical_not(points_equal).sum(axis=1)
    counts = np.bincount(distances, minlength=X.shape[1] + 1)
    return counts/float(X.shape[0]), np.mean(distances), np.std(distances)


def mean_reciprocal_rank(X, Y, indices):
    ''' Computes the mean reciprocal rank of the correct match
    Assumes that X[n] should be closest to Y[n]
    Uses hamming distance

    :parameters:
        - X : np.ndarray, shape=(n_examples, n_features)
            Data matrix in X modality
        - Y : np.ndarray, shape=(n_examples, n_features)
            Data matrix in Y modality
        - indices : np.ndarray
            Denotes which rows to use in MRR calculation

    :returns:
        - mrr_pessimist : float
            Mean reciprocal rank, where ties are resolved pessimistically
            That is, rank = # of distances <= dist(X[:, n], Y[:, n])
        - mrr_optimist : float
            Mean reciprocal rank, where ties are resolved optimistically
            That is, rank = # of distances < dist(X[:, n], Y[:, n]) + 1
    '''
    # Compute distances between each codeword and each other codeword
    distance_matrix = scipy.spatial.distance.cdist(X, Y, metric='hamming')
    # Rank is the number of distances smaller than the correct distance, as
    # specified by the indices arg
    n_le = distance_matrix.T <= distance_matrix[np.arange(X.shape[0]), indices]
    n_lt = distance_matrix.T < distance_matrix[np.arange(X.shape[0]), indices]
    return (np.mean(1./n_le.sum(axis=0)),
            np.mean(1./(n_lt.sum(axis=0) + 1)))


def save_model(param_list, output_file):
    '''
    Write out a pickle file of a hashing network

    :parameters:
        - param_list : list of np.ndarray
            A list of values, per layer, of the parameters of the network
        - output_file : str
            Path to write the file to
    '''
    with open(output_file, 'wb') as f:
        pickle.dump(param_list, f)


def load_model(param_list, batch_size):
    '''
    Create a hashing network based on a list of per-layer parameters

    :parameters:
        - param_list : list of np.ndarray
            A list of values, per layer, of the parameters of the network
        - batch_size : int
            The input batch size, which cannot be inferred from the parameter
            shapes

    :returns:
        - layers : list of lasagne.layers.Layer
            List of all layers in the network
    '''
    # Layers in the hashing network
    layers = []
    # Start with input layer
    layers.append(lasagne.layers.InputLayer(
        shape=(batch_size, param_list[-2].shape[0])))
    # Add each hidden layer recursively
    for W, b in zip(param_list[-2::-2], param_list[::-2]):
        layers.append(lasagne.layers.DenseLayer(
            layers[-1], num_units=W.shape[1],
            nonlinearity=lasagne.nonlinearities.tanh))
        # Set the param value according to the input
        layers[-1].W.set_value(W.astype(theano.config.floatX))
        layers[-1].b.set_value(b.astype(theano.config.floatX))
    return layers
