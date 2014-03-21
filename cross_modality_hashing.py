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
     + .5*T.sum(T.maximum(0, m - T.sqrt(T.sum((X_net.layers_n[-1].output - Y_net.layers_n[-1].output)**2, axis=0)))**2)

# List of update steps for each parameter
updates = []
learning_rate = 1e-3
# Just gradient descent on cost
for param in X_net.params + Y_net.params:
    updates.append((param, param - learning_rate*T.grad(cost, param)))

# Function for optimizing the neural net parameters, by minimizing cost
train = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input, m], cost, updates=updates)
# Functions for actually computing the output of each neural net
X_eval = theano.function([X_p_input], X_net.layers_p[-1].output)
Y_eval = theano.function([Y_p_input], Y_net.layers_p[-1].output)

# <codecell>

if __name__=='__main__':
    import glob
    import matplotlib.pyplot as plt
    from IPython import display
    
    def shingle(x, stacks):
        ''' Shingle a matrix column-wise '''
        return np.vstack([x[:, n:(x.shape[1] - stacks + n)] for n in xrange(stacks)])

    def load_data(directory):
        ''' Load in all chroma matrices and piano rolls and output them as separate matrices '''
        X = []
        Y = []
        for chroma_filename in glob.glob(directory + '*-msd.npy'):
            X += [shingle(np.load(chroma_filename), 4)]
            piano_roll_filename = chroma_filename.replace('msd', 'midi')
            Y += [shingle(np.load(piano_roll_filename), 4)]
        return np.hstack(X), np.hstack(Y)
        
    def get_next_batch(X, Y, batch_size, n_iter):
        ''' Fast (hopefully) random mini batch generator '''
        n_batches = int(np.floor(X.shape[1]/batch_size))
        current_batch = n_batches
        for n in xrange(n_iter):
            if current_batch >= n_batches:
                positive_shuffle = np.random.permutation(X.shape[1])
                negative_shuffle = np.random.permutation(X.shape[1])
                X_p = X[:, positive_shuffle]
                Y_p = Y[:, positive_shuffle]
                X_n = X[:, negative_shuffle]
                Y_n = Y[:, np.roll(negative_shuffle, 1)]
                current_batch = 0
            batch = np.r_[current_batch*batch_size:(current_batch + 1)*batch_size]
            yield X_p[:, batch], Y_p[:, batch], X_n[:, batch], Y_n[:, batch]
            current_batch += 1

    def standardize(X):
        ''' Standardize the rows of a data matrix X '''
        std = np.std(X, axis=1).reshape(-1, 1)
        return (X - np.mean(X, axis=1).reshape(-1, 1))/(std + (std == 0))
      
    def hashes_used(X):
        ''' Get the number of unique hashes actually used '''
        return np.unique(np.sum(2**np.arange(X.shape[0]).reshape(-1, 1)*X, axis=0)).shape[0]

    def count_errors(X, Y):
        ''' Computes the number of correctly encoded codeworks and the number of bit errors made '''
        points_equal = (X == Y)
        return np.all(points_equal, axis=0).sum(), np.logical_not(points_equal).sum(), hashes_used(X), hashes_used(Y)
    
    # Load in the data
    X, Y = load_data('data/hash_dataset/')
    
    # Standardize
    X = standardize(X)
    Y = standardize(Y)
    
    # Split into .8 train/test indices
    train_indices = np.random.sample(X.shape[1]) < .8
    X_train = np.array(X[:, train_indices])
    Y_train = np.array(Y[:, train_indices])
    validate_indices = np.logical_not(train_indices)
    X_validate = np.array(X[:, validate_indices])
    Y_validate = np.array(Y[:, validate_indices])
    
    # Randomly select some data vectors to plot every so often
    plot_indices_train = np.random.randint(0, X_train.shape[1], 20)
    plot_indices_validate = np.random.randint(0, X_validate.shape[1], 20)
    
    # Value of m_{XY} to use
    m_val = 4
    
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(get_next_batch(X_train, Y_train, 100, int(1e8))):
        current_cost = train(X_p, X_n, Y_p, Y_n, m_val)
        # Every so many, print the cost and plot some diagnostic figures
        if not n % 1000:
            display.clear_output()
            print "Itertation {}".format(n)
            print "Cost: {}".format(current_cost)
            
            for name, X_set, Y_set, plot_indices in [('Train', X_train, Y_train, plot_indices_train),
                                                     ('Validate', X_validate, Y_validate, plot_indices_validate)]:
                print 
                print name
                X_output = X_eval(X_set)
                Y_output = Y_eval(Y_set)
                correct, errors, hashes_X, hashes_Y = count_errors(Y_output > 0, 
                                                                   X_output > 0)
                print "  {} of {} vectors hashed correctly".format(correct, X_set.shape[1])
                print "  {} of {} bits incorrect".format(errors, X_set.shape[1]*n_bits)
                print "  {}, {} of {} possible hashes used".format(hashes_X, hashes_Y, 2**n_bits)
                plt.figure(figsize=(18, 2))
                
                for n, image in enumerate([Y_output[:, plot_indices],
                                           X_output[:, plot_indices],
                                           Y_output[:, plot_indices] > 0,
                                           X_output[:, plot_indices] > 0,
                                           np.not_equal(X_output[:, plot_indices] > 0, Y_output[:, plot_indices] > 0)]):
                    plt.subplot(1, 5, n + 1)
                    plt.imshow(image, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
            plt.show()

