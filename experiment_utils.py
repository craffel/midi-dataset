''' Shared utility functions for downsampled hash sequence experiments. '''

import lasagne
import numpy as np
import os
import collections
import deepdish
import traceback
import functools
import glob
import sys
import simple_spearmint

N_BITS = 32


def run_trial(params, data_directory, train_function):
    '''
    Train a network given the task and hyperparameters and return the result.

    Parameters
    ----------
    params : dict
        Dictionary of model hyperparameters
    data_directory : str
        Path to training/validation set directory.  Should have two
        subdirectories, one called 'train' and one called 'validate', each of
        which contain subdirectories called 'h5', which contain training files
        created by `deepdish`.
    train_function : callable
        This function will be called with the constructed network, training
        data, and hyperparameters to create a model.

    Returns
    -------
    best_objective : float
        Lowest objective value achieved.
    best_epoch : dict
        Statistics about the epoch during which the lowest objective value was
        achieved.
    best_params : dict
        Parameters of the model for the best-objective epoch.
    '''
    # Load in data as dictionary of dictionaries
    data = {'X': collections.defaultdict(list),
            'Y': collections.defaultdict(list)}
    for set in ['train', 'validate']:
        for f in glob.glob(os.path.join(data_directory, set, 'h5', '*.h5')):
            for k, v in deepdish.io.load(f).items():
                data[k][set].append(v)

    # Build networks
    layers = {}
    for network in ['X', 'Y']:
        # Get # of features (last dimension) from first training sequence
        input_shape = (None, 1, None, data[network]['train'][0].shape[-1])
        # Get training set statistics for standardization
        input_mean = np.mean(
            np.concatenate(data[network]['train'], axis=1), axis=1)
        input_std = np.std(
            np.concatenate(data[network]['train'], axis=1), axis=1)
        # Choose network structure based on network param
        if params['network'] == 'big_filter':
            build_network = build_network_big_filter
        elif params['network'] == 'small_filters':
            build_network = build_network_small_filters
        else:
            raise ValueError('Unknown network {}'.format(params['network']))
        layers[network] = build_network(
            input_shape, input_mean, input_std,
            params['downsample_frequency'], params['dropout'])

    # Create updates-creating function
    updates_function = functools.partial(
        lasagne.updates.rmsprop, learning_rate=params['learning_rate'],
        rho=params['momentum'])

    print ',\n'.join(['\t{} : {}'.format(k, v) for k, v in params.items()])
    # Create a list of epochs
    epochs = []
    # Keep track of lowest objective found so far
    best_objective = np.inf
    try:
        for epoch in train_function(
                data, layers, params['negative_importance'],
                params['negative_threshold'], params['entropy_importance'],
                updates_function):
            # Stop training if a nan training cost is encountered
            if not np.isfinite(epoch['train_cost']):
                break
            epochs.append(epoch)
            if epoch['validate_objective'] < best_objective:
                best_objective = epoch['validate_objective']
                best_epoch = epoch
                best_model = {
                    'X': lasagne.layers.get_all_param_values(layers['X']),
                    'Y': lasagne.layers.get_all_param_values(layers['Y'])}
            print "{}: {}, ".format(epoch['iteration'],
                                    epoch['validate_objective']),
            sys.stdout.flush()
    # If there was an error while training, report it to whetlab
    except Exception:
        print "ERROR: "
        print traceback.format_exc()
        return np.nan, {}, {}
    print
    # Check that all training costs were not NaN; return NaN if any were.
    success = np.all([np.isfinite(e['train_cost']) for e in epochs])
    if np.isinf(best_objective) or len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return np.nan, {}, {}
    else:
        for k, v in best_epoch.items():
            print "\t{:>35} | {}".format(k, v)
        print
        return best_objective, best_epoch, best_model


def parameter_search(space, trial_directory, model_directory, data_directory,
                     train_function):
    '''
    Run parameter optimization given some train function, writing out results

    Parameters
    ----------
    space : dict
        Hyperparameter space (in the format used by `simple_spearmint`) to
        optimize over
    trial_directory : str
        Directory where parameter optimization trial results will be written
    model_directory : str
        Directory where the best-performing model will be written
    data_directory : str
        Path to training/validation set directory.  Should have two
        subdirectories, one called 'train' and one called 'validate', each of
        which contain subdirectories called 'h5', which contain training files
        created by `deepdish`.
    train_function : callable
        This function will be called with the constructed network, training
        data, and hyperparameters to create a model.
    '''
    # Create parameter trials directory if it doesn't exist
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)
    # Same for model directory
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Create SimpleSpearmint suggester instance
    ss = simple_spearmint.SimpleSpearmint(space)

    # Load in previous results for "warm start"
    for trial_file in glob.glob(os.path.join(trial_directory, '*.h5')):
        trial = deepdish.io.load(trial_file)
        ss.update(trial['hyperparameters'], trial['best_objective'])

    # Run parameter optimization forever
    while True:
        # Get a new suggestion
        suggestion = ss.suggest()
        # Train a network with these hyperparameters
        best_objective, best_epoch, best_model = run_trial(
            suggestion, data_directory, train_function)
        # Update spearmint on the result
        ss.update(suggestion, best_objective)
        # Write out a result file
        trial_filename = ','.join('{}={}'.format(k, v)
                                  for k, v in suggestion.items()) + '.h5'
        deepdish.io.save(
            os.path.join(trial_directory, trial_filename),
            {'hyperparameters': suggestion, 'best_objective': best_objective,
             'best_epoch': best_epoch})
        # Also write out the entire model when the objective is the smallest
        # We don't want to write all models; they are > 100MB each
        if (not np.isnan(best_objective) and
                best_objective == np.min(ss.objective_values)):
            deepdish.io.save(
                os.path.join(model_directory, 'best_model.h5'), best_model)


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


def get_valid_matches(pair_file, score_threshold, diagnostics_path):
    '''
    Reads in a CSV file listing text-matched pairs, finds the pairs
    corresponding to the MSD, and then returns only those pairs which were
    successfully aligned.

    Parameters
    ----------
    pair_file : str
        Full path to a CSV file listing text-matched pairs (generated by
        scripts/text_match_datasets.py)

    score_threshold : float
        Alignments will only be considered correct if their normalized DTW
        score was above this threshold.

    diagnostics_path : str
        Full path to where alignment diagnostics files have been written

    Returns
    -------
    pairs : dict
        Mapping from MIDI MD5s to list of MSD IDs which match it
    '''
    midi_msd_mapping = collections.defaultdict(list)
    with open(pair_file) as f:
        for line in f.readlines():
            midi_md5, dataset, msd_id = line.strip().split(',')
            # The pairs.csv files will include pairs from all datasets
            # Only grab those for the MSD
            if dataset == 'msd':
                # Only include if the alignment was successful
                alignment_file = os.path.join(
                    diagnostics_path, 'msd_{}_{}.h5'.format(msd_id, midi_md5))
                if os.path.exists(alignment_file):
                    diagnostics = deepdish.io.load(alignment_file)
                    if diagnostics['score'] > score_threshold:
                        midi_msd_mapping[midi_md5].append(msd_id)
    return dict(midi_msd_mapping)
