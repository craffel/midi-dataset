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
    for chroma_filename in glob.glob(directory + '*-msd.npy'):
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

# Load in the data
X_train, Y_train, X_validate, Y_validate = load_data('../data/hash_dataset/')

# Standardize
X_mean, X_std = standardize(X_train)
X_train = (X_train - X_mean)/X_std
X_validate = (X_validate - X_mean)/X_std
Y_mean, Y_std = standardize(Y_train)
Y_train = (Y_train - Y_mean)/Y_std
Y_validate = (Y_validate - Y_mean)/Y_std

# Dimensionality of hamming space
n_bits = 16
# Number of layers in each network
n_layers = 3

# Compute layer sizes.  Middle layers are nextpow2(input size)
layer_sizes_x = [X_train.shape[0]] + [int(2**np.ceil(np.log2(X_train.shape[0])))]*(n_layers - 1) + [n_bits]
layer_sizes_y = [Y_train.shape[0]] + [int(2**np.ceil(np.log2(Y_train.shape[0])))]*(n_layers - 1) + [n_bits]
hasher = cross_modality_hashing.SiameseNet(layer_sizes_x, layer_sizes_y)

# First neural net, for chroma vectors
X_p_input = T.matrix('X_p_input')
X_n_input = T.matrix('X_n_input')
# Second neural net, for MIDI piano roll
Y_p_input = T.matrix('Y_p_input')
Y_n_input = T.matrix('Y_n_input')
# Hyperparameters
alpha_XY = T.scalar('alpha_XY')
m_XY = T.scalar('m_XY')
alpha_X = T.scalar('alpha_X')
m_X = T.scalar('m_X')
alpha_Y = T.scalar('alpha_Y')
m_Y = T.scalar('m_Y')
learning_rate = 1e-4
momentum = .9
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

# Randomly select some data vectors to plot every so often
plot_indices_train = np.random.choice(X_train.shape[1], 20, False)
plot_indices_validate = np.random.choice(X_validate.shape[1], 20, False)

# Value of m_{XY} to use
alpha_XY_val = 1
m_XY_val = 8
alpha_X_val = 1
m_X_val = 8
alpha_Y_val = 1
m_Y_val = 8

# Maximum number of iterations to run
n_iter = int(1e8)
# Store the cost at each iteration
costs = np.zeros(n_iter)

try:
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(get_next_batch(X_train, Y_train, 10, n_iter)):
        costs[n] = train(X_p, X_n, Y_p, Y_n, alpha_XY_val, m_XY_val, alpha_X_val, m_X_val, alpha_Y_val, m_Y_val)
        # Every so many iterations, print the cost and plot some diagnostic figures
        if not n % 5000:
            display.clear_output()
            print "Iteration {}".format(n)
            print "Cost: {}".format(costs[n])

            # Get accuracy and diagnostic figures for both train and validation sets
            for name, X_set, Y_set, plot_indices in [('Train', X_train, Y_train, plot_indices_train),
                                                     ('Validate', X_validate, Y_validate, plot_indices_validate)]:
                print
                print name
                # Get the network output for this dataset
                X_output = hasher.X_net.output(X_set).eval()
                Y_output = hasher.Y_net.output(Y_set).eval()
                N = X_set.shape[1]
                # Compute and display metrics on the resulting hashes
                correct, in_class_mean, in_class_std = statistics(X_output > 0, Y_output > 0)
                collisions, out_of_class_mean, out_of_class_std = statistics(X_output[:, np.random.permutation(N)] > 0,
                                                                             Y_output > 0)
                print "  {}/{} = {:.3f}% vectors hashed correctly".format(correct, N, correct/(1.*N)*100)
                print "  {:.3f} +/- {:.3f} average in-class distance".format(in_class_mean, in_class_std)
                print "  {}/{} = {:.3f}% hash collisions".format(collisions, N, collisions/(1.*N)*100)
                print "  {:.3f} +/- {:.3f} average out-of-class distance".format(out_of_class_mean, out_of_class_std)
                print "  Entropy: {:.4f}, {:.4f}".format(hash_entropy(X_output > 0), hash_entropy(Y_output > 0), 2**n_bits)
                print

                plt.figure(figsize=(18, 2))
                # Show images of each networks output, binaraized and nonbinarized, and the error
                for n, image in enumerate([Y_output[:, plot_indices],
                                           X_output[:, plot_indices],
                                           Y_output[:, plot_indices] > 0,
                                           X_output[:, plot_indices] > 0,
                                           np.not_equal(X_output[:, plot_indices] > 0, Y_output[:, plot_indices] > 0)]):
                    plt.subplot(1, 5, n + 1)
                    plt.imshow(image, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
            plt.show()
except KeyboardInterrupt:
    costs = costs[:n]
    plt.figure(figsize=(12, 12))
    plt.plot(costs)

