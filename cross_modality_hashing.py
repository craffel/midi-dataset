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
chroma_p = T.matrix('chroma_p')
chroma_n = T.matrix('chroma_n')
# Number and size of layers chosen somewhat arbitrarily
chroma_net = MLP_two_inputs(chroma_p, chroma_n, [48, 64, 64, n_bits])
# Second neural net, for MIDI piano roll
piano_roll_p = T.matrix('piano_roll_p')
piano_roll_n = T.matrix('piano_roll_n')
piano_roll_net = MLP_two_inputs(piano_roll_p, piano_roll_n, [192, 256, 256, n_bits])

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
cost = .5*T.sum((chroma_net.layers_p[-1].output - piano_roll_net.layers_p[-1].output)**2) \
     + .5*T.sum(T.maximum(0, m - T.sqrt(T.sum((chroma_net.layers_n[-1].output - piano_roll_net.layers_n[-1].output)**2, axis=0)))**2)

# List of update steps for each parameter
updates = []
learning_rate = 1e-3
# Just gradient descent on cost
for param in chroma_net.params + piano_roll_net.params:
    updates.append((param, param - learning_rate*T.grad(cost, param)))

# Function for optimizing the neural net parameters, by minimizing cost
train = theano.function([chroma_p, chroma_n, piano_roll_p, piano_roll_n, m], cost, updates=updates)
# Functions for actually computing the output of each neural net
chroma_eval = theano.function([chroma_p], chroma_net.layers_p[-1].output)
piano_roll_eval = theano.function([piano_roll_p], piano_roll_net.layers_p[-1].output)

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
        for chroma_filename in glob.glob(directory + '*-chroma.npy'):
            X += [shingle(np.load(chroma_filename), 4)]
            piano_roll_filename = chroma_filename.replace('chroma', 'piano_roll')
            Y += [shingle(np.load(piano_roll_filename)[36:84, :], 4)]
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
    chroma_data_p, piano_roll_data_p = load_data('../data/theano_test/')
    
    # Standardize
    chroma_data_p = standardize(chroma_data_p)
    piano_roll_data_p = standardize(piano_roll_data_p)
    
    # Randomly select some data vectors to plot every so often
    plot_indices = np.random.randint(0, piano_roll_data_p.shape[1], 20)
    
    # Value of m_{XY} to use
    m_val = 4
    
    for n, (X_p, Y_p, X_n, Y_n) in enumerate(get_next_batch(chroma_data_p, piano_roll_data_p, 100, int(1e8))):
        current_cost = train(X_p, X_n, Y_p, Y_n, m_val)
        # Every so many, print the cost and plot some diagnostic figures
        if not n % 1000:
            display.clear_output()
            correct, errors, hashes_X, hashes_Y = count_errors(piano_roll_eval(piano_roll_data_p) > 0, 
                                                               chroma_eval(chroma_data_p) > 0)
            print "Itertation {}".format(n)
            print "Cost: {}".format(current_cost)
            print "{} of {} vectors hashed correctly".format(correct, piano_roll_data_p.shape[1])
            print "{} of {} bits incorrect".format(errors, piano_roll_data_p.shape[1]*n_bits)
            print "{}, {} of {} possible hashes used".format(hashes_X, hashes_Y, 2**n_bits)
            plt.figure(figsize=(18, 2))
            plt.subplot(151)
            plt.imshow(piano_roll_eval(piano_roll_data_p[:, plot_indices]), aspect='auto', interpolation='nearest')
            plt.subplot(152)
            plt.imshow(chroma_eval(chroma_data_p[:, plot_indices]), aspect='auto', interpolation='nearest', )
            plt.subplot(153)
            plt.imshow(piano_roll_eval(piano_roll_data_p[:, plot_indices]) > 0,
                       aspect='auto', interpolation='nearest', cmap=plt.cm.cool)
            plt.subplot(154)
            plt.imshow(chroma_eval(chroma_data_p[:, plot_indices]) > 0,
                       aspect='auto', interpolation='nearest', cmap=plt.cm.cool)
            plt.subplot(155)
            plt.imshow(np.not_equal(chroma_eval(chroma_data_p[:, plot_indices]) > 0,
                                    piano_roll_eval(piano_roll_data_p[:, plot_indices]) > 0),
                       aspect='auto', interpolation='nearest', cmap=plt.cm.cool)
            plt.show()

