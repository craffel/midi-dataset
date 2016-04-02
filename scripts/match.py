'''
Code for matching one or more MIDI files to the MSD
Call it like this:
    python match.py ../data/unique_midi/mid/0/\*.mid
'''

import dhs
import numpy as np
import pretty_midi
import sys
import theano
import traceback
import os
import deepdish
import msgpack
import djitw
import glob
import lasagne
sys.path.append('..')
import feature_extraction
import whoosh_search
import experiment_utils

RESULTS_PATH = '../results'
DATA_PATH = '../data'
# Keep the MSD entries which had the N smallest embedding distance
TOP_EMBEDDINGS = 10000
# Keep the MSD entries which had the N smallest hash sequence distance
TOP_SEQUENCES = 250
# Parameters for hash sequence DTW
GULLY = .9
PENALTY = 15
# If a MIDI file is bigger than this, skip it
MAX_FRAMES = 10000
# Skip any hash sequences shorter than this
MIN_SEQUENCE_LENGTH = 30


def match_one_midi(midi_filename, embed_fn, hash_fn, msd_embeddings,
                   msd_sequences, msd_feature_paths, msd_ids, output_filename):
    """
    Match one MIDI file to the million song dataset by computing its CQT,
    pruning by matching its embedding, re-pruning by matching its downsampled
    hash sequence, and finally doing DTW on CQTs on the remaining entries.

    Parameters
    ----------
    midi_filename : str
        Path to a MIDI file to match to the MSD
    embed_fn : function
        Function which takes in a CQT and produces a fixed-length embedding
    hash_fn : function
        Function which takes in a CQT and produces a sequence of binary vectors
    msd_embeddings : np.ndarray
        (# MSD entries x embedding dimension) matrix of all embeddings for all
        entries from the MSD
    msd_sequences : list of np.ndarray
        List of binary vector sequences (represented as ints) for all MSD
        entries
    msd_feature_paths` : list of str
        Path to feature files (containing CQT) for each MSD entry
    msd_ids : list of str
        MSD ID of each corresponding entry in the above lists
    output_filename : str
        Where to write the results file, which includes the DTW scores for all
        of the non-pruned MSD entries
    """
    # Try to compute a CQT for the MIDI file
    try:
        m = pretty_midi.PrettyMIDI(midi_filename)
    except Exception as e:
        print 'Could not parse {}: {}'.format(
            os.path.split(midi_filename)[1], traceback.format_exc(e))
        return
    try:
        midi_gram = feature_extraction.midi_cqt(m)
    except Exception as e:
        print "Error creating CQT for {}: {}".format(
            os.path.split(midi_filename)[1], traceback.format_exc(e))
        return
    # Skip this file if the MIDI gram is very long, to avoid memory issues
    if midi_gram.shape[0] > MAX_FRAMES:
        return
    # Compute the embedding of the CQT
    midi_embedding = embed_fn(midi_gram.reshape(1, 1, *midi_gram.shape))
    # Get the distance between the MIDI embedding and all MSD entries
    embedding_distances = np.sum((msd_embeddings - midi_embedding)**2, axis=1)
    # Get the indices of MSD entries sorted by their embedded distance to the
    # query MIDI embedding.
    embedding_matches = np.argsort(embedding_distances)
    # Get the top N matches
    embedding_matches = embedding_matches[:TOP_EMBEDDINGS]
    # Compute the hash sequence
    midi_hash_sequence = hash_fn(midi_gram.reshape(1, 1, *midi_gram.shape))
    # Convert to sequence of integers
    midi_hash_sequence = dhs.vectors_to_ints(midi_hash_sequence > 0)
    midi_hash_sequence = midi_hash_sequence.astype(np.uint32)
    # Match this hash sequence to MSD sequences
    hash_matches, _, _ = dhs.match_one_sequence(
        midi_hash_sequence, msd_sequences, GULLY, PENALTY, True,
        embedding_matches)
    # Get the top N matches
    hash_matches = hash_matches[:TOP_SEQUENCES]
    # List for storing final match information
    matches = []
    # Perform DTW matching for each non-pruned MSD entry
    for match in hash_matches:
        # Construct path to pre-computed audio CQT path
        audio_features_filename = os.path.join(msd_feature_paths[match])
        try:
            audio_features = deepdish.io.load(audio_features_filename)
        except Exception as e:
            print "Error loading CQT for {}: {}".format(
                os.path.split(midi_filename)[1], traceback.format_exc(e))
            continue
        # Check that the distance matrix will not be too big before computing
        size = midi_gram.shape[0] * audio_features['gram'].shape[0]
        # If > 1 GB, skip
        if (size * 64 / 8e9 > 2):
            print (
                "Distance matrix for {} and {} would be {} GB because the "
                "CQTs have shape {} and {}".format(
                    os.path.split(audio_features_filename)[1],
                    os.path.split(midi_filename)[1],
                    size * 64 / 8e9, audio_features['gram'].shape[0],
                    midi_gram.shape[0]))
            continue
        # Get distance matrix
        distance_matrix = 1 - np.dot(midi_gram, audio_features['gram'].T)
        # Non-diagonal additive path penalty is the median of the sim mtx
        add_pen = np.median(distance_matrix)
        # Get best path through matrix
        aligned_midi_indices, aligned_audio_indices, score = djitw.dtw(
            distance_matrix, gully=.96, additive_penalty=add_pen,
            inplace=False)
        # Normalize score by path length
        score /= float(len(aligned_midi_indices))
        # Normalize score by score by mean sim matrix value within path chunk
        score /= distance_matrix[
            aligned_midi_indices.min():aligned_midi_indices.max(),
            aligned_audio_indices.min():aligned_audio_indices.max()].mean()
        # The confidence score is a normalized DTW distance, which
        # approximately follows in the range [.5, 1.] with .5 meaning a very
        # good alignment.  This maps the scores from [0., 1.] where 1. means a
        # very good alignment.
        score = np.clip(2 * (1 - score), 0, 1)
        matches.append([msd_ids[match], score])
    # Write out the result
    with open(output_filename, 'wb') as f:
        msgpack.dump(matches, f)


if __name__ == '__main__':
    # Create PSE network hashing function
    # Load in all parameter optimization trials
    trial_files = glob.glob(os.path.join(
        RESULTS_PATH, 'pse_parameter_trials', '*.h5'))
    trials = [deepdish.io.load(f) for f in trial_files]
    # Get the hyperparameters for the trial with the lowest objective value
    best_trial = sorted(trials, key=lambda t: t['best_objective'])[0]
    hyperparameters = best_trial['hyperparameters']
    # Load in the pre-trained parameters for the best performing models
    network_params = deepdish.io.load(
        os.path.join(RESULTS_PATH, 'pse_model', 'best_model.h5'))
    # Construct the network according to best-trial hyperparameters
    if hyperparameters['network'] == 'pse_big_filter':
        build_network = experiment_utils.build_pse_net_big_filter
    elif hyperparameters['network'] == 'pse_small_filters':
        build_network = experiment_utils.build_pse_net_small_filters
    # PSE trials do not have this hyperparameter by default, so as in
    # experiment_utils.run_trial, we must set the default value
    hyperparameters['downsample_frequency'] = hyperparameters.get(
        'downsample_frequency', True)
    layers = build_network(
        (None, 1, None, feature_extraction.N_NOTES),
        # We will supply placeholders here but load in the values below
        np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
        np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
        hyperparameters['downsample_frequency'],
        hyperparameters['n_attention'], hyperparameters['n_conv'])
    # Load in network parameter values
    lasagne.layers.set_all_param_values(
        layers[-1], network_params['X'])
    # Compile function for computing the output of the network
    embed_fn = theano.function(
        [layers[0].input_var],
        lasagne.layers.get_output(layers[-1], deterministic=True))

    # Create DHS network hashing function
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
        layers[-1], network_params['X'])
    # Compile function for computing the output of the network
    hash_fn = theano.function(
        [layers[0].input_var],
        lasagne.layers.get_output(layers[-1], deterministic=True))

    # Load in list of MSD entries
    msd_index = whoosh_search.get_whoosh_index(
        os.path.join(DATA_PATH, 'msd', 'index'))
    with msd_index.searcher() as searcher:
        msd_list = list(searcher.documents())

    # Load MSD embeddings
    msd_embedding_datas = experiment_utils.load_precomputed_data(
        msd_list, os.path.join(RESULTS_PATH, 'pse_msd_embeddings'))
    # Load in hash sequences (and metadata) for all MSD entries
    msd_sequence_datas = experiment_utils.load_precomputed_data(
        msd_list, os.path.join(RESULTS_PATH, 'dhs_msd_hash_sequences'))
    # Ignore entries with very short sequences
    msd_embedding_datas = [
        e for d, e in zip(msd_sequence_datas, msd_embedding_datas)
        if len(d['hash_sequence']) > MIN_SEQUENCE_LENGTH]
    msd_sequence_datas = [d for d in msd_sequence_datas
                          if len(d['hash_sequence']) > MIN_SEQUENCE_LENGTH]
    # Create a big matrix of the embeddings
    msd_embeddings = np.concatenate(
        [d['embedding'] for d in msd_embedding_datas], axis=0)
    # Construct paths to each feature file
    msd_feature_paths = [
        os.path.join(DATA_PATH, 'msd', 'h5', d['path']) + '.h5'
        for d in msd_sequence_datas]
    # Extract all sequences
    msd_sequences = [d['hash_sequence'] for d in msd_sequence_datas]
    # Extract all IDs
    msd_ids = [d['id'] for d in msd_sequence_datas]

    # We no longer need these big objects, keeping them around wastes memory.
    # Running processes in parallel requires more memory, so delete them.
    del msd_list
    del msd_embedding_datas
    del msd_sequence_datas
    del network_params

    # Use the first argument as the glob of things to look for (including a
    # single file)
    midi_filenames = glob.glob(sys.argv[1])
    # Construct output filenames
    output_filenames = [
        os.path.join(RESULTS_PATH, 'unique_midi_matched',
                     os.path.split(f)[1][0],
                     os.path.splitext(os.path.split(f)[1])[0] + '.mpk')
        for f in midi_filenames]
    # Construct intermediate directories for all output filenames
    for output_filename in output_filenames:
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])

    # Match each of the files
    for midi_fname, output_fname in zip(midi_filenames, output_filenames):
        match_one_midi(midi_fname, embed_fn, hash_fn, msd_embeddings,
                       msd_sequences, msd_feature_paths, msd_ids, output_fname)
