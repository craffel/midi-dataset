"""
Precompute hash sequences for all MSD entries and piano rolls of all MIDI files
from clean_midi
"""
# We'll use msgpack for I/O.  It seems to be fastest, and is widely supported.
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import deepdish
import lasagne
import glob
import os
import numpy as np
import traceback
import sys
import dhs
import theano
import pretty_midi
import librosa
sys.path.append(os.path.join('..', '..'))
import feature_extraction
import whoosh_search
import experiment_utils

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'

if __name__ == '__main__':
    # Load in all parameter optimization trials
    trial_files = glob.glob(os.path.join(
        RESULTS_PATH, 'dhs_piano_parameter_trials', '*.h5'))
    trials = [deepdish.io.load(f) for f in trial_files]
    # Get the hyperparameters for the trial with the lowest objective value
    best_trial = sorted(trials, key=lambda t: t['best_objective'])[0]
    hyperparameters = best_trial['hyperparameters']
    # Load in the pre-trained parameters for the best performing models
    network_params = deepdish.io.load(
        os.path.join(RESULTS_PATH, 'dhs_piano_model', 'best_model.h5'))
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
                if dataset == 'msd':
                    # Construct CQT h5 file path from file index entry
                    filename = os.path.join(
                        DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                    # Load in CQT
                    gram = deepdish.io.load(filename)['gram']
                else:
                    # Construct MIDI file path file file index entry
                    filename = os.path.join(
                        DATA_PATH, dataset, 'mid', entry['path'] + '.mid')
                    # Construct PrettyMIDI object
                    pm = pretty_midi.PrettyMIDI(filename)
                    # Create list of frames whose spacing matches CQT spacing
                    max_frame = int(pm.get_end_time() *
                                    feature_extraction.MIDI_FS /
                                    feature_extraction.MIDI_HOP)
                    midi_frame_times = librosa.frames_to_time(
                        np.arange(max_frame), sr=feature_extraction.MIDI_FS,
                        hop_length=feature_extraction.MIDI_HOP)
                    # Retrieve piano roll
                    gram = pm.get_piano_roll(times=midi_frame_times)
                    # Only utilize the same notes which are used in the CQT
                    gram = gram[feature_extraction.NOTE_START:
                                (feature_extraction.NOTE_START +
                                 feature_extraction.N_NOTES)]
                    # Transpose so that the first dimension is time
                    gram = gram.T
                    # L2 normalize columns
                    gram = librosa.util.normalize(gram, norm=2, axis=1)
                    # Use float32 for Theano
                    gram = gram.astype(np.float32)
                # Compute (real-valued, vector) hash sequence
                hash_sequence = compute_output(
                    gram.reshape(1, 1, *gram.shape))
                # Convert to sequence of integers
                hash_sequence = dhs.vectors_to_ints(hash_sequence > 0)
                hash_sequence = hash_sequence.astype(np.uint32)
                # Construct output path to the same location in
                # RESULTS_PATH/dhs_piano_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH,
                    'dhs_piano_{}_hash_sequences'.format(dataset),
                    entry['path'] + '.mpk')
                # Construct intermediate subdirectories if they don't exist
                if not os.path.exists(os.path.split(output_file)[0]):
                    os.makedirs(os.path.split(output_file)[0])
                # Save result, along with the index entry for convenience
                with open(output_file, 'wb') as f:
                    f.write(msgpack.packb(
                        dict(hash_sequence=hash_sequence, **entry)))
            except Exception:
                print "Error processing : {}".format(filename)
                print traceback.format_exc()
