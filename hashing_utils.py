'''
Utilities for cross-modality hashing experiments, including data loading and
statistics
'''
import numpy as np
import theano
import glob
import os
import scipy.spatial
import pickle
import lasagne


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


def load_data(directory, shingle_size=4):
    ''' Load in all chroma matrices and piano rolls and output them as
    separate matrices

    :parameters:
        - directory : string
            Path to datast directory.  Expects .npz files which contain ['X']
            and ['Y'] features for each modality.
        - shingle_size : int
            Number of copies of each column to stack

    :returns:
        - X : list
            List of np.ndarrays of X modality features, each entry is a
            different track
        - Y : list
            List of np.ndarrays of Y modality features, each entry is a
            different track
    '''
    X = []
    Y = []
    for filename in glob.glob(os.path.join(directory, '*.npz')):
        data = np.load(filename)
        X.append(shingle(data['X'], shingle_size))
        Y.append(shingle(data['Y'], shingle_size))
    return ([np.array(x, dtype=theano.config.floatX, order='C') for x in X],
            [np.array(y, dtype=theano.config.floatX, order='C') for y in Y])


def standardize(X):
    ''' Return column vectors to standardize X, via (X - X_mean)/X_std

    :parameters:
        - X : np.ndarray, shape=(n_features, n_examples)
            Data matrix

    :returns:
        - X_mean : np.ndarray, shape=(n_features, 1)
            Mean column vector
        - X_std : np.ndarray, shape=(n_features, 1)
            Standard deviation column vector
    '''
    std = np.std(X, axis=0)
    return np.mean(X, axis=0), std + (std == 0)


def train_validate_split(X, Y, split=.9):
    ''' Splits a dataset into train and validate sets randomly and standardizes
    them

    :parameters:
        - X : list
            List of np.ndarray data matrices in one modality
        - Y : list
            List of np.ndarray data matrices in another modality
        - split : float
            Proportion of data to keep in training set, default .9

    :returns:
        - X_train : np.ndarray
            Vertically stacked data matrices in X training set
        - Y_train : np.ndarray
            Vertically stacked data matrices in Y training set
        - X_validate : np.ndarray
            Vertically stacked data matrices in X validation set
        - Y_validate : np.ndarray
            Vertically stacked data matrices in Y validation set
    '''
    # Start as lists, will vstack later
    X_train = []
    Y_train = []
    X_validate = []
    Y_validate = []
    # Randomly add data matrices to X and Y lists
    for x, y in zip(X, Y):
        if np.random.rand() < split:
            X_train.append(x)
            Y_train.append(y)
        else:
            X_validate.append(x)
            Y_validate.append(y)
    # Turn lists into large data matrices
    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)
    X_validate = np.vstack(X_validate)
    Y_validate = np.vstack(Y_validate)
    # Standardize using statistics fom training set
    X_mean, X_std = standardize(X_train)
    X_train = (X_train - X_mean)/X_std
    X_validate = (X_validate - X_mean)/X_std
    Y_mean, Y_std = standardize(Y_train)
    Y_train = (Y_train - Y_mean)/Y_std
    Y_validate = (Y_validate - Y_mean)/Y_std

    return X_train, Y_train, X_validate, Y_validate


def get_next_batch(X, Y, batch_size, n_iter):
    ''' Randomly generates positive and negative example minibatches

    :parameters:
        - X : np.ndarray, shape=(n_features, n_examples)
            Data matrix in one modality
        - y : np.ndarray, shape=(n_features, n_examples)
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
        - X : np.ndarray, shape=(n_bits, n_examples)
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
        - X : np.ndarray, shape=(n_features, n_examples)
            Data matrix of X modality
        - Y : np.ndarray, shape=(n_features, n_examples)
            Codeword matrix of Y modality

    :returns:
        - n_correct : int
            Number of examples correctly hashed
        - mean_distance : float
            Mean of distances between corresponding codewords
        - std_distance : float
            Std of distances between corresponding codewords
    '''
    points_equal = (X == Y)
    return (np.all(points_equal, axis=1).sum(),
            np.mean(np.logical_not(points_equal).sum(axis=1)),
            np.std(np.logical_not(points_equal).sum(axis=1)))


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
