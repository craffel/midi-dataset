"""
Precompute hash sequences for all MSD entries and MIDI files from clean_midi
"""
# We'll use msgpack for I/O.  It seems to be fastest, and is widely supported.
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import deepdish
import lasagne
import glob
import os
import sys
sys.path.append(os.path.join('..', '..'))
import feature_extraction
import experiment_utils
import theano
import numpy as np
import dhs
import traceback
import whoosh_search

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'

if __name__ == '__main__':
    # Load in all parameter optimization trials
    trial_files = glob.glob(os.path.join(
        RESULTS_PATH, 'dhs_parameter_trials', '*.h5'))
    trials = [deepdish.io.load(f) for f in trial_files]
    # Get the hyperparameters for the trial with the lowest objective value
    best_trial = sorted(trials, key=lambda t: t['best_objective'])[0]
    hyperparameters = best_trial['hyperparameters']
    # Load in the pre-trained parameters for the best performing models
    network_params = deepdish.io.load(
        os.path.join(RESULTS_PATH, 'dhs_model', 'best_model.h5'))
    # Build networks and output-computing functions
    for dataset, network in zip(['clean_midi', 'msd'], ['X', 'Y']):
        # Get file list from whoosh index
        index = whoosh_search.get_whoosh_index(
            os.path.join(DATA_PATH, dataset, 'index'))
        with index.searcher() as searcher:
            file_list = list(searcher.documents())
        # We only need to hash MIDI files from the dev or test sets
        if dataset == 'clean_midi':
            md5s = [line.split(',')[0]
                    for pair_file in [
                        os.path.join(RESULTS_PATH, 'dev_pairs.csv'),
                        os.path.join(RESULTS_PATH, 'test_pairs.csv')]
                    for line in open(pair_file)
                    if line.split(',')[1] == 'msd']
            file_list = [e for e in file_list if e['id'] in md5s]
        # Construct the network according to best-trial hyperparameters
        if hyperparameters['network'] == 'dhs_big_filter':
            build_network = experiment_utils.build_dhs_net_big_filter
        elif hyperparameters['network'] == 'dhs_small_filters':
            build_network = experiment_utils.build_dhs_net_small_filters
        layers = build_network(
            (None, 1, None, feature_extraction.N_NOTES),
            # We will supply placeholders here but load in the values below
            np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
            np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
            hyperparameters['downsample_frequency'])
        # Load in network parameter values
        lasagne.layers.set_all_param_values(
            layers[-1], network_params[network])
        # Compile function for computing the output of the network
        compute_output = theano.function(
            [layers[0].input_var],
            lasagne.layers.get_output(layers[-1], deterministic=True))
        # Load in CQTs and write out downsampled hash sequences
        for entry in file_list:
            try:
                # Construct CQT h5 file path from file index entry
                h5_file = os.path.join(
                    DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                # Load in CQT
                gram = deepdish.io.load(h5_file)['gram']
                # Compute (real-valued, vector) hash sequence
                hash_sequence = compute_output(
                    gram.reshape(1, 1, *gram.shape))
                # Convert to sequence of integers
                hash_sequence = dhs.vectors_to_ints(hash_sequence > 0)
                hash_sequence = hash_sequence.astype(np.uint32)
                # Construct output path to the same location in
                # RESULTS_PATH/dhs_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH, 'dhs_{}_hash_sequences'.format(dataset),
                    entry['path'] + '.mpk')
                # Construct intermediate subdirectories if they don't exist
                if not os.path.exists(os.path.split(output_file)[0]):
                    os.makedirs(os.path.split(output_file)[0])
                # Save result, along with the index entry for convenience
                with open(output_file, 'wb') as f:
                    f.write(msgpack.packb(
                        dict(hash_sequence=hash_sequence, **entry)))
            except Exception:
                print "Error processing : {}".format(h5_file)
                print traceback.format_exc()
