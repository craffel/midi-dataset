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
    def __init__(self, n_input, n_output, W=None, b=None, activation=T.tanh):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        Input:
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
        self.activation = activation

        # Easy-to-access parameter list
        self.params = [self.W, self.b]
        
    def output(self, x):
        '''
        Compute this layer's output given an input
        
        Input:
            x - Theano symbolic variable for layer input
        Output:
            output - Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        return (lin_output if self.activation is None else self.activation(lin_output))

# <codecell>

class MLP(object):
    def __init__(self, layer_sizes=None, Ws=None, bs=None, activations=None):
        '''
        Multi-layer perceptron

        Input:
            layer_sizes - List-like of layer sizes, len n_layers + 1, includes input and output dimensionality
                Default None, which means retrieve layer sizes from W
            Ws - List-like of weight matrices, len n_layers, where Ws[n] is layer_sizes[n + 1] x layer_sizes[n]
                Default None, which means initialize randomly
            bs - List-like of biases, len n_layers, where bs[n] is layer_sizes[n + 1]
                Default None, which means initialize randomly
            activations - List of length n_layers of activation function for each layer
                Default None, which means all layers are tanh
        '''
        # Check that we received layer sizes or weight matrices + bias vectors
        if layer_sizes is None and Ws is None:
            raise ValueError('Either layer_sizes or Ws must not be None')

        # Initialize lists of layers
        self.layers = []

        # Populate layer sizes if none was provided
        if layer_sizes is None:
            layer_sizes = []
            # Each layer size is the input size of each mixing matrix
            for W in Ws:
                layer_sizes.append(W.shape[1])
            # plus the output size of the last layer
            layer_sizes.append(Ws[-1].shape[0])

        # Make a list of Nones if Ws and bs are None
        if Ws is None:
            Ws = [None]*(len(layer_sizes) - 1)
        if bs is None:
            bs = [None]*(len(layer_sizes) - 1)

        # All activations are tanh if none was provided
        if activations is None:
            activations = [T.tanh]*(len(layer_sizes) - 1)
            
        # Construct the layers
        for n, (n_input, n_output, W, b, activation) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], Ws, bs, activations)):
            self.layers.append(Layer(n_input, n_output, W, b, activation=activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self, x):
        '''
        Compute the MLP's output given an input
        
        Input:
            x - Theano symbolic variable for MLP input
        Output:
            output - x passed through the net
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

# <codecell>

class SiameseNet(object):
    def __init__(self, layer_sizes_x, layer_sizes_y, activations_x=None, activations_y=None):
        '''
        "Siamese" neural net, which takes inputs from two modalities and maps them to a common modality
        
        Input:
            layer_sizes_x - List-like of layer sizes for "x" net, len n_layers + 1, includes input and output dimensionality
            layer_sizes_y - List-like of layer sizes for "y" net, len n_layers + 1, includes input and output dimensionality
            activations_x - List of length n_layers of activation function for each layer in "x" net
                Default None, which means all layers are tanh
            activations_x - List of length n_layers of activation function for each layer in "y" net
                Default None, which means all layers are tanh
        '''
        # Create each network
        self.X_net = MLP(layer_sizes_x, activations=activations_x)
        self.Y_net = MLP(layer_sizes_y, activations=activations_y)
        # Concatenate list of parameters
        self.params = self.X_net.params + self.Y_net.params
        
    def cross_modality_cost(self, x_p, x_n, y_p, y_n, alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_y):
        '''
        Compute the cost of encoding a set of positive and negative inputs from both modalities
        
        Input:
            x_p - Theano symbolic variable of positive examples from modality "x"
            x_n - Theano symbolic variable of negative examples from modality "x"
            y_p - Theano symbolic variable of positive examples from modality "y"
            y_n - Theano symbolic variable of negative examples from modality "y"
            alpha_XY - Theano symbolic variable for scaling parameter for cross-modality negative example cost
            m_XY - Theano symbolic variable for cross-modality negative example threshold
            alpha_X - Theano symbolic variable for scaling parameter for X-modality negative example cost
            m_X - Theano symbolic variable for Y-modality negative example threshold
            alpha_Y - Theano symbolic variable for scaling parameter for Y-modality negative example cost
            m_Y - Theano symbolic variable for Y-modality negative example threshold
        Output:
            cost - cost, given these parameters and data
        '''
        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.sum((self.X_net.output(x_p) - self.Y_net.output(y_p))**2)
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*T.sum(T.maximum(0, 4*m_XY - T.sum((self.X_net.output(x_n) - self.Y_net.output(y_n))**2, axis=0)))
        # Thresholded, scaled cost of x-modality negative examples
        cost_x = alpha_X*T.sum(T.maximum(0, 4*m_X - T.sum((self.X_net.output(x_p) - self.X_net.output(x_n))**2, axis=0)))
        # Thresholded, scaled cost of y-modality negative examples
        cost_y = alpha_Y*T.sum(T.maximum(0, 4*m_Y - T.sum((self.Y_net.output(y_p) - self.Y_net.output(y_n))**2, axis=0)))
        # Return sum of these costs
        return cost_p + cost_n + cost_x + cost_y

# <codecell>

def gradient_updates(cost, params, learning_rate):
    '''
    Compute updates for gradient descent over some parameters.
    
    Input:
        cost - Theano cost function to minimize
        params - Parameters to compute gradient against
        learning_rate - GD learning rate
    Output:
        updates - list of updates, per-parameter
    '''
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        updates.append((param, param - learning_rate*T.grad(cost, param)))
    return updates

# <codecell>

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    Input:
        cost - Theano cost function to minimize
        params - Parameters to compute gradient against
        learning_rate - GD learning rate
        momentum - GD momentum
    Output:
        updates - list of updates, per-parameter
    '''
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        mparam = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*mparam))
        updates.append((mparam, mparam*momentum + (1. - momentum)*T.grad(cost, param)))
    return updates

# <markdowncell>

# Cost is given by
# $$
# \mathcal{L}_{XY} = \frac{1}{2} \sum_{(x, y) \in \mathcal{P}_{XY}} \|\xi(x) - \eta(y)\|_2^2
# + \frac{1}{2} \sum_{(x, y) \in \mathcal{N}_{XY}} \max \{0, m_{XY} - \|\xi(x) - \eta(y)\|_2\}^2
# $$
# where $(x, y) \in \mathcal{P}_{XY}$ denotes that $x$ and $y$ are in the same class and $(x, y) \in \mathcal{N}_{XY}$ denotes that $x$ and $y$ are in different classes, $\xi$ is the function of one neural network and $\eta$ is the function of the other and $m_{XY}$ is the hinge loss threshold.

# <codecell>

if __name__=='__main__':
    import glob
    import matplotlib.pyplot as plt
    from IPython import display

    def shingle(x, stacks):
        ''' Shingle a matrix column-wise '''
        return np.vstack([x[:, n:(x.shape[1] - stacks + n)] for n in xrange(stacks)])

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

    def statistics(X, Y):
        ''' Computes the number of correctly encoded codeworks and the number of bit errors made '''
        points_equal = (X == Y)
        return np.all(points_equal, axis=0).sum(), \
               np.mean(np.logical_not(points_equal).sum(axis=0)), \
               np.std(np.logical_not(points_equal).sum(axis=0))

    # Load in the data
    X_train, Y_train, X_validate, Y_validate = load_data('data/hash_dataset/')

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
    hasher = SiameseNet(layer_sizes_x, layer_sizes_y)

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
                            updates=gradient_updates_momentum(hasher_cost,
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
                    collisions, out_of_class_mean, out_of_class_std = statistics(X_output[:, np.random.permutation(N)] > 0, Y_output > 0)
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

