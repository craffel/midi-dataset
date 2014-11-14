# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

''' Functions for mapping data in different modalities to a common Hamming space '''

# <codecell>

import numpy as np
import theano.tensor as T
import theano
import hashing_utils
import collections

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
        
    def cross_modality_cost(self, x_p, x_n, y_p, y_n, alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y):
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

def train_cross_modality_hasher(X_train, Y_train, X_validate, Y_validate,
                                hidden_layer_sizes_X, hidden_layer_sizes_Y,
                                alpha_XY_val, m_XY_val, alpha_X_val, m_X_val,
                                alpha_Y_val, m_Y_val, n_bits,
                                learning_rate=1e-4, momentum=.9, batch_size=10,
                                epoch_size=1000, initial_patience=10000,
                                improvement_threshold=0.99, patience_increase=1.2,
                                max_iter=200000, mrr_samples=None):
    ''' Utility function for training a siamese net for cross-modality hashing
    So many parameters.
    Assumes, e.g., X_train[:, n] should be mapped close to Y_train[:, m] only when n == m
    
    :parameters:
        - X_train : np.ndarray, shape=(n_features, n_train_examples)
            Training data in X modality
        - Y_train : np.ndarray, shape=(n_features, n_train_examples)
            Training data in Y modality
        - X_validate : np.ndarray, shape=(n_features, n_validate_examples)
            Validation data in X modality
        - Y_validate : np.ndarray, shape=(n_features, n_validate_examples)
            Validation data in Y modality
        - hidden_layer_sizes_X : list-like
            Size of each hidden layer in X network.
            Number of layers = len(hidden_layer_sizes_X) + 1
        - hidden_layer_sizes_Y : list-like
            Size of each hidden layer in Y network.
            Number of layers = len(hidden_layer_sizes_Y) + 1
        - alpha_XY_val : float
            Scaling parameter for cross-modality negative example cost
        - m_XY_val : int
            Cross-modality negative example threshold
        - alpha_X_val : float
            Scaling parameter for X-modality negative example cost
        - m_X_val : int
            Y-modality negative example threshold
        - alpha_Y_val : float
            Scaling parameter for Y-modality negative example cost
        - m_Y_val : int
            Y-modality negative example threshold
        - n_bits : int
            Number of bits in the output representation
        - learning_rate : float
            SGD learning rate, default 1e-4
        - momemntum : float
            SGD momentum, default .9
        - batch_size : int
            Mini-batch size, default 10
        - epoch_size : int
            Number of mini-batches per epoch, default 1000
        - initial_patience : int
            Always train on at least this many batches, default 10000
        - improvement_threshold : float
            Validation cost must decrease by this factor to increase patience, default 0.99
        - patience_increase : float
            Amount to increase patience when validation cost has decreased, default 1.2
        - max_iter : int
            Maximum number of batches to train on, default 200000
        - mrr_samples : np.ndarray
            Indices of samples in the validation set over which to compute mean reciprocal rank.
            Default None, which means use entire validation set.

    :returns:
        - epochs : list
            List of epoch dicts, which contains scores computed at each epoch
        - parameters : list
            List of NN parameters after each epoch
    '''
    # First neural net, for X modality
    X_p_input = T.matrix('X_p_input')
    X_n_input = T.matrix('X_n_input')
    # Second neural net, for MIDI piano roll
    Y_p_input = T.matrix('Y_p_input')
    Y_n_input = T.matrix('Y_n_input')
    # Symbolic hyperparameters
    alpha_XY = T.scalar('alpha_XY')
    m_XY = T.scalar('m_XY')
    alpha_X = T.scalar('alpha_X')
    m_X = T.scalar('m_X')
    alpha_Y = T.scalar('alpha_Y')
    m_Y = T.scalar('m_Y')
    # Create siamese neural net hasher
    layer_sizes_X = [X_train.shape[0]] + hidden_layer_sizes_X + [n_bits]
    layer_sizes_Y = [Y_train.shape[0]] + hidden_layer_sizes_Y + [n_bits]
    hasher = SiameseNet(layer_sizes_X, layer_sizes_Y)
    
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
    # Compute cost without trianing
    cost = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input,
                            alpha_XY, m_XY, alpha_X, m_X, alpha_Y, m_Y], hasher_cost)

    # Keep track of the patience - we will always increase the patience once
    patience = initial_patience/patience_increase
    current_validate_cost = np.inf
    
    # Functions for computing the neural net output on the train and validation sets
    X_train_output = hasher.X_net.output(X_train)
    Y_train_output = hasher.Y_net.output(Y_train)
    X_validate_output = hasher.X_net.output(X_validate)
    Y_validate_output = hasher.Y_net.output(Y_validate)
    
    # A list of epoch result dicts, one per epoch
    epochs = []
    # A list of parameter settings at each epoch
    parameters = []
    
    # Create fixed negative example validation set
    X_validate_n = X_validate[:, np.random.permutation(X_validate.shape[1])]
    Y_validate_n = Y_validate[:, np.random.permutation(Y_validate.shape[1])]

    for n, (X_p, Y_p, X_n, Y_n) in enumerate(hashing_utils.get_next_batch(X_train, Y_train, batch_size, max_iter)):
        train_cost = train(X_p, X_n, Y_p, Y_n, alpha_XY_val, m_XY_val, alpha_X_val, m_X_val, alpha_Y_val, m_Y_val)
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Store current SGD cost
            epoch_result['train_cost'] = train_cost
            # Also compute validate cost (more stable)
            epoch_result['validate_cost'] = cost(X_validate, X_validate_n, Y_validate, Y_validate_n, alpha_XY_val,
                                                 m_XY_val, alpha_X_val, m_X_val, alpha_Y_val, m_Y_val)
            
            # Get accuracy and diagnostic figures for both train and validation sets
            for name, X_output, Y_output in [('train', X_train_output.eval(), Y_train_output.eval()),
                                             ('validate', X_validate_output.eval(), Y_validate_output.eval())]:
                N = X_output.shape[1]
                # Compute on the resulting hashes
                correct, in_class_mean, in_class_std = hashing_utils.statistics(X_output > 0, Y_output > 0)
                collisions, out_mean, out_std = hashing_utils.statistics(X_output[:, np.random.permutation(N)] > 0,
                                                                         
                                                                                           Y_output > 0)
                epoch_result[name + '_accuracy'] = correct/float(N)
                epoch_result[name + '_in_class_distance_mean'] = in_class_mean
                epoch_result[name + '_in_class_distance_std'] = in_class_std
                epoch_result[name + '_collisions'] = collisions/float(N)
                epoch_result[name + '_out_of_class_distance_mean'] = out_mean
                epoch_result[name + '_out_of_class_distance_std'] = out_std
                epoch_result[name + '_hash_entropy_X'] = hashing_utils.hash_entropy(X_output > 0)
                epoch_result[name + '_hash_entropy_Y'] = hashing_utils.hash_entropy(Y_output > 0)
            if epoch_result['validate_cost'] < current_validate_cost:
                if epoch_result['validate_cost'] < improvement_threshold*current_validate_cost:
                    patience *= patience_increase
                    print " ... increasing patience to {} because {} < {}*{}".format(patience,
                                                                                     epoch_result['validate_cost'],
                                                                                     improvement_threshold,
                                                                                     current_validate_cost)
                current_validate_cost = epoch_result['validate_cost']
            # Only compute MRR on validate
            mrr_pessimist, mrr_optimist = hashing_utils.mean_reciprocal_rank(X_output[:, mrr_samples] > 0,
                                                                             Y_output > 0,
                                                                             mrr_samples)
            epoch_result['validate_mrr_pessimist'] = mrr_pessimist
            epoch_result['validate_mrr_optimist'] = mrr_optimist
            
            # Store scores and statistics for this epoch
            epochs.append(epoch_result)
            
            # Get current parameter settings
            current_parameters = collections.OrderedDict()
            for net_name, net in zip(['X_net', 'Y_net'], [hasher.X_net, hasher.Y_net]):
                for n, layer in enumerate(net.layers):
                    for parameter in layer.params:
                        current_parameters["{}_layer_{}_{}".format(net_name, n, parameter.name)] = parameter.get_value()
            # Store parameters for this epoch
            parameters.append(current_parameters)
            
            print '    patience : {}'.format(patience)
            print '    current_validation_cost : {}'.format(current_validate_cost)
            for k, v in epoch_result.items():
                print '    {} : {}'.format(k, round(v, 3))
            print
            
        if n > patience:
            break
    
    return epochs, parameters

