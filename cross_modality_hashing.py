# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

''' Functions for mapping data in different modalities to a common Hamming space '''

# <codecell>

import numpy as np
import theano.tensor as T
import theano

# <codecell>

class Layer(object):
    def __init__(self, x, n_input, n_output, W=None, b=None, activation=T.tanh):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
        
        Input:
            x - Theano symbolic variable for layer input
            n_input - number of input nodes
            n_output - number of output nodes
            W - Mixing matrix, default None which means initialize randomly
            b - Bias vector, default None which means initialize to ones
            activation - nonlinear activation function, default tanh
        '''
        # Randomly initialize W
        if W is None:
            # Tanh is best initialized to values between +/- sqrt(6/(n_nodes))
            W_values = np.asarray(np.random.uniform( -np.sqrt(6./(n_input + n_output)), 
                                                     np.sqrt(6./(n_input + n_output)),
                                                     (n_output, n_input)), 
                                                     dtype=theano.config.floatX)
            # Sigmoid activation uses +/- 4*sqrt(6/(n_nodes))
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            # Create theano shared variable for W
            W = theano.shared(value=W_values, name='W', borrow=True)
        # Initialize b to zeros
        if b is None:
            b = theano.shared(value=np.ones((n_output, 1), dtype=theano.config.floatX),
                              name='b', borrow=True, broadcastable=(False, True))
        
        self.W = W
        self.b = b
        
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        self.output = (lin_output if activation is None else activation(lin_output))
        # Easy-to-access parameter list
        self.params = [self.W, self.b]

# <codecell>

class MLP_two_inputs(object):
    def __init__(self, x_p, x_n, layer_sizes, activations=None):
        '''
        MLP class from which it is convenient to get the output for two data matrices at the same time
        
        Input:
            x_p - data matrix 1            
            x_n - data matrix 2
            layer_sizes - List of length N of layer sizes, includes input and output dimensionality.
                Resulting MLP will have N-1 layers.
            activations - List of length N-1 of activation function for each layer, default tanh
        '''
        # Initialize lists of layers
        self.layers_p = []
        self.layers_n = []
        # All activations are tanh if none was provided
        if activations is None:
            activations = [T.tanh]*(len(layer_sizes) - 1)
        # Construct the layers
        for n, (n_input, n_output, activation) in enumerate( zip( layer_sizes[:-1], layer_sizes[1:], activations ) ):
            # For first layer, input is x
            if n == 0:
                # Need to create two MLPs...
                self.layers_p += [Layer(x_p, n_input, n_output, activation=activation)]
                # where weight matrices and bias vectors are shared
                self.layers_n += [Layer(x_n, n_input, n_output, W=self.layers_p[0].W, b=self.layers_p[0].b, activation=activation)]
            # Otherwise, input is previous layer's output
            else:
                # Layer's input is previous layer's output
                self.layers_p += [Layer(self.layers_p[n-1].output, n_input, n_output, activation=activation)]                
                # As above
                self.layers_n += [Layer(self.layers_n[n-1].output,
                                        n_input, n_output,
                                        W=self.layers_p[n].W,
                                        b=self.layers_p[n].b,
                                        activation=activation)]
        # Combine parameters from all layers
        self.params = []
        for layer in self.layers_p:
            self.params += layer.params

# <codecell>

# Dimensionality of hamming space
n_bits = 8
# First neural net, for chroma vectors
X_p_input = T.matrix('X_p_input')
X_n_input = T.matrix('X_n_input')
# Number and size of layers chosen somewhat arbitrarily
X_net = MLP_two_inputs(X_p_input, X_n_input, [100, 128, 128, n_bits])
# Second neural net, for MIDI piano roll
Y_p_input = T.matrix('Y_p_input')
Y_n_input = T.matrix('Y_n_input')
Y_net = MLP_two_inputs(Y_p_input, Y_n_input, [192, 256, 256, n_bits])

# <markdowncell>

# Cost is given by
# $$
# \mathcal{L}_{XY} = \frac{1}{2} \sum_{(x, y) \in \mathcal{P}_{XY}} \|\xi(x) - \eta(y)\|_2^2
# + \frac{1}{2} \sum_{(x, y) \in \mathcal{N}_{XY}} \max \{0, m_{XY} - \|\xi(x) - \eta(y)\|_2\}^2
# $$
# where $(x, y) \in \mathcal{P}_{XY}$ denotes that $x$ and $y$ are in the same class and $(x, y) \in \mathcal{N}_{XY}$ denotes that $x$ and $y$ are in different classes, $\xi$ is the function of one neural network and $\eta$ is the function of the other and $m_{XY}$ is the hinge loss threshold.

# <codecell>

# Threshold parameter
m = T.scalar('m')
# Compute cost function as described above
cost = .5*T.sum((X_net.layers_p[-1].output - Y_net.layers_p[-1].output)**2) \
#     + .5*T.sum(T.maximum(0, m - T.sum((X_net.layers_n[-1].output - Y_net.layers_n[-1].output)**2, axis=0)))
     + .5*T.sum(T.maximum(0, m - T.sqrt(T.sum((X_net.layers_n[-1].output - Y_net.layers_n[-1].output)**2, axis=0)))**2)

# List of update steps for each parameter
updates = []
learning_rate = 1e-4
# Just gradient descent on cost
for param in X_net.params + Y_net.params:
    updates.append((param, param - learning_rate*T.grad(cost, param)))

# Function for optimizing the neural net parameters, by minimizing cost
train = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input, m], cost, updates=updates)
# Functions for actually computing the output of each neural net
X_eval = theano.function([X_p_input], X_net.layers_p[-1].output)
Y_eval = theano.function([Y_p_input], Y_net.layers_p[-1].output)

# <codecell>

PLOT=False
FLOATX=np.float32

# <codecell>

if __name__=='__main__':
    import glob
    if PLOT:
        import matplotlib.pyplot as plt
        from IPython import display
    
    def shingle(x, stacks):
        ''' Shingle a matrix column-wise '''
        return np.vstack([x[:, n:(x.shape[1] - stacks + n)] for n in xrange(stacks)])

    def load_data(directory, shingle_size=4, train_validate_split=.8):
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
        return np.array(np.hstack(X_train), dtype=FLOATX), \
               np.array(np.hstack(Y_train), dtype=FLOATX), \
               np.array(np.hstack(X_validate), dtype=FLOATX), \
               np.array(np.hstack(Y_validate), dtype=FLOATX)

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
                X_n = np.array(X[:, np.mod(negative_shuffle + 2*np.random.randint(0, 2, N) - 1, N)])
                #X_n = np.array(X[:, np.random.permutation(N)])
                Y_n = np.array(Y[:, negative_shuffle])
                current_batch = 0
            batch = np.r_[current_batch*batch_size:(current_batch + 1)*batch_size]
            yield X_p[:, batch], Y_p[:, batch], X_n[:, batch], Y_n[:, batch]
            current_batch += 1

    def standardize(X):
        ''' Return column vectors to standardize X, via (X - X_mean)/X_std '''
        std = np.std(X, axis=1).reshape(-1, 1)
        return np.mean(X, axis=1).reshape(-1, 1), std + (std == 0)
      
    def hash_entropy(X):
        ''' Get the entropy of the histogram of hashes (want this to be close to n_bits) '''
        bit_values = np.sum(2**np.arange(X.shape[0]).reshape(-1, 1)*X, axis=0)
        counts, _ = np.histogram(bit_values, np.arange(2**X.shape[0]))
        counts = counts/float(counts.sum())
        return -np.sum(counts*np.log2(counts + 1e-100))
        
    def count_errors(X, Y):
        ''' Computes the number of correctly encoded codeworks and the number of bit errors made '''
        points_equal = (X == Y)
        return np.all(points_equal, axis=0).sum(), np.logical_not(points_equal).sum(), hash_entropy(X), hash_entropy(Y)
    
    # Load in the data
    X_train, Y_train, X_validate, Y_validate = load_data('data/hash_dataset/')
    
    # Standardize
    X_mean, X_std = standardize(X_train)
    X_train = (X_train - X_mean)/X_std
    X_validate = (X_validate - X_mean)/X_std
    Y_mean, Y_std = standardize(Y_train)
    Y_train = (Y_train - Y_mean)/Y_std
    Y_validate = (Y_validate - Y_mean)/Y_std
    
    # Randomly select some data vectors to plot every so often
    plot_indices_train = np.random.randint(0, X_train.shape[1], 20)
    plot_indices_validate = np.random.randint(0, X_validate.shape[1], 20)
    
    # Value of m_{XY} to use
    m_val = 5
    
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(get_next_batch(X_train, Y_train, 100, int(1e8))):
        current_cost = train(X_p, X_n, Y_p, Y_n, m_val)
        # Every so many iterations, print the cost and plot some diagnostic figures
        if not n % 1000:
            if PLOT:
                display.clear_output()
            print "Iteration {}".format(n)
            print "Cost: {}".format(current_cost)
            
            # Get accuracy and diagnostic figures for both train and validation sets
            for name, X_set, Y_set, plot_indices in [('Train', X_train, Y_train, plot_indices_train),
                                                     ('Validate', X_validate, Y_validate, plot_indices_validate)]:
                print 
                print name
                # Get the network output for this dataset
                X_output = X_eval(X_set)
                Y_output = Y_eval(Y_set)
                # Compute and display metrics on the resulting hashes
                correct, errors, hash_entropy_X, hash_entropy_Y = count_errors(Y_output > 0, X_output > 0)
                N = X_set.shape[1]
                print "  {}/{} = {:.3f}% vectors hashed correctly".format(correct, N, correct/(1.*N)*100)
                print "  {}/{} = {:.3f}% bits incorrect".format(errors, N*n_bits, errors/(1.*N*n_bits)*100)
                print "  Entropy: {:.4f}, {:.4f}".format(hash_entropy_X, hash_entropy_Y, 2**n_bits)
                
                if PLOT:
                    plt.figure(figsize=(18, 2))
                    # Show images of each networks output, binaraized and nonbinarized, and the error
                    for n, image in enumerate([Y_output[:, plot_indices],
                                               X_output[:, plot_indices],
                                               Y_output[:, plot_indices] > 0,
                                               X_output[:, plot_indices] > 0,
                                               np.not_equal(X_output[:, plot_indices] > 0, Y_output[:, plot_indices] > 0)]):
                        plt.subplot(1, 5, n + 1)
                        plt.imshow(image, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
            if PLOT:
                plt.show()

