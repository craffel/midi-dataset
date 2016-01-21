''' Shared utility functions for downsampled hash sequence experiments. '''

import lasagne
import numpy as np

N_BITS = 16


def build_network_small_filters(input_shape, input_mean, input_std,
                                downsample_frequency, dropout, n_bits=N_BITS):
    '''
    Construct a list of layers of a network which has three groups of two 3x3
    convolutional layers followed by a max-pooling layer.

    Parameters
    ----------
    input_shape : tuple
        In what shape will data be supplied to the network?
    input_mean : np.ndarray
        Training set mean, to standardize inputs with.
    input_std : np.ndarray
        Training set standard deviation, to standardize inputs with.
    downsample_frequency : bool
        Whether to max-pool over frequency
    dropout : bool
        Should dropout be applied between fully-connected layers?
    n_bits : int
        Output dimensionality

    Returns
    -------
    layers : list
        List of layer instances for this network.
    '''
    layers = [lasagne.layers.InputLayer(shape=input_shape)]
    # Utilize training set statistics to standardize all inputs
    layers.append(lasagne.layers.standardize(
        layers[-1], input_mean, input_std, shared_axes=(0, 2)))
    # Construct the pooling size based on whether we pool over frequency
    if downsample_frequency:
        pool_size = (2, 2)
    else:
        pool_size = (2, 1)
    # Add three groups of 2x 3x3 convolutional layers followed by a pool layer
    filter_size = (3, 3)
    for num_filters in [16, 32, 64]:
        n_l = num_filters*np.prod(filter_size)
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], pool_size, ignore_border=False))
    # A dense layer will treat any dimensions after the first as feature
    # dimensions, but the third dimension is really a timestep dimension.
    # We can only squash adjacent dimensions with a ReshapeLayer, so we
    # need to place the time stpe dimension after the batch dimension
    layers.append(lasagne.layers.DimshuffleLayer(
        layers[-1], (0, 2, 1, 3)))
    conv_output_shape = layers[-1].output_shape
    # Reshape to (n_batch*n_time_steps, n_conv_output_features)
    layers.append(lasagne.layers.ReshapeLayer(
        layers[-1], (-1, conv_output_shape[2]*conv_output_shape[3])))
    # Add dense hidden layers and optionally dropout
    for hidden_layer_size in [2048, 2048]:
        layers.append(lasagne.layers.DenseLayer(
            layers[-1], num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify))
        if dropout:
            layers.append(lasagne.layers.DropoutLayer(layers[-1], .5))
    # Add output layer
    layers.append(lasagne.layers.DenseLayer(
        layers[-1], num_units=n_bits,
        nonlinearity=lasagne.nonlinearities.tanh))

    return layers


def build_network_big_filter(input_shape, input_mean, input_std,
                             downsample_frequency, dropout, n_bits=N_BITS):
    '''
    Construct a list of layers of a network which has a ``big'' 5x12 input
    filter and two 3x3 convolutional layers, all followed by max-pooling
    layers.

    Parameters
    ----------
    input_shape : tuple
        In what shape will data be supplied to the network?
    input_mean : np.ndarray
        Training set mean, to standardize inputs with.
    input_std : np.ndarray
        Training set standard deviation, to standardize inputs with.
    downsample_frequency : bool
        Whether to max-pool over frequency
    dropout : bool
        Should dropout be applied between fully-connected layers?
    n_bits : int
        Output dimensionality

    Returns
    -------
    layers : list
        List of layer instances for this network.
    '''
    layers = [lasagne.layers.InputLayer(shape=input_shape)]
    # Utilize training set statistics to standardize all inputs
    layers.append(lasagne.layers.standardize(
        layers[-1], input_mean, input_std, shared_axes=(0, 2)))
    # Construct the pooling size based on whether we pool over frequency
    if downsample_frequency:
        pool_size = (2, 2)
    else:
        pool_size = (2, 1)
    # The first convolutional layer has filter size (5, 12), and Lasagne
    # doesn't allow same-mode convolutions with even filter sizes.  So, we need
    # to explicitly use a pad layer.
    filter_size = (5, 12)
    num_filters = 16
    layers.append(lasagne.layers.PadLayer(
        layers[-1], width=((int(np.ceil((filter_size[0] - 1) / 2.)),
                           int(np.floor((filter_size[0] - 1) / 2.))),
                           (int(np.ceil((filter_size[1] - 1) / 2.)),
                           int(np.floor((filter_size[1] - 1) / 2.))))))
    # We will initialize weights to \sqrt{2/n_l}
    n_l = num_filters*np.prod(filter_size)
    layers.append(lasagne.layers.Conv2DLayer(
        layers[-1], stride=(1, 1), num_filters=num_filters,
        filter_size=filter_size,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(np.sqrt(2./n_l))))
    layers.append(lasagne.layers.MaxPool2DLayer(
        layers[-1], pool_size, ignore_border=False))
    # Add two 3x3 convolutional layers with 32 and 64 filter,s and pool layers
    filter_size = (3, 3)
    for num_filters in [32, 64]:
        n_l = num_filters*np.prod(filter_size)
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], pool_size, ignore_border=False))
    # A dense layer will treat any dimensions after the first as feature
    # dimensions, but the third dimension is really a timestep dimension.
    # We can only squash adjacent dimensions with a ReshapeLayer, so we
    # need to place the time stpe dimension after the batch dimension
    layers.append(lasagne.layers.DimshuffleLayer(
        layers[-1], (0, 2, 1, 3)))
    conv_output_shape = layers[-1].output_shape
    # Reshape to (n_batch*n_time_steps, n_conv_output_features)
    layers.append(lasagne.layers.ReshapeLayer(
        layers[-1], (-1, conv_output_shape[2]*conv_output_shape[3])))
    # Add dense hidden layers and optionally dropout
    for hidden_layer_size in [2048, 2048]:
        layers.append(lasagne.layers.DenseLayer(
            layers[-1], num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify))
        if dropout:
            layers.append(lasagne.layers.DropoutLayer(layers[-1], .5))
    # Add output layer
    layers.append(lasagne.layers.DenseLayer(
        layers[-1], num_units=n_bits,
        nonlinearity=lasagne.nonlinearities.tanh))

    return layers
