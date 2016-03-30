'''
Match entries in the clean MIDI subset to the Million Song Dataset
'''
import os
import sys
sys.path.append(os.path.join('..', '..'))
import whoosh_search
import experiment_utils
import os
import joblib
import deepdish

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'
# Should we use the test set or development set?
SPLIT = 'dev'
# DTW parameters
GULLY = .9
PENALTY = 15
# Ignore all hash sequences below this length (they cause issues)
MIN_SEQUENCE_LENGTH = 30
# A DP score above this means the alignment is correct
SCORE_THRESHOLD = .5

if __name__ == '__main__':
    # Load in list of MSD entries
    msd_index = whoosh_search.get_whoosh_index(
        os.path.join(DATA_PATH, 'msd', 'index'))
    with msd_index.searcher() as searcher:
        msd_list = list(searcher.documents())

    # Load in list of MSD entries
    midi_index = whoosh_search.get_whoosh_index(
        os.path.join(DATA_PATH, 'clean_midi', 'index'))
    with midi_index.searcher() as searcher:
        midi_list = list(searcher.documents())

    # Load in hash sequences (and metadata) for all MSD entries
    msd_data = experiment_utils.load_precomputed_data(
        msd_list, os.path.join(RESULTS_PATH, 'dhs_msd_hash_sequences'))
    # Ignore very short sequences
    msd_data = [d for d in msd_data
                if len(d['hash_sequence']) > MIN_SEQUENCE_LENGTH]

    # Get a list of valid MIDI-MSD match pairs
    midi_msd_mapping = experiment_utils.get_valid_matches(
        os.path.join(RESULTS_PATH, '{}_pairs.csv'.format(SPLIT)),
        SCORE_THRESHOLD,
        os.path.join(RESULTS_PATH, 'clean_midi_aligned', 'h5'))

    midi_datas, midi_index_mapping = experiment_utils.load_valid_midi_datas(
        midi_msd_mapping, msd_data, midi_list,
        os.path.join(RESULTS_PATH, 'dhs_clean_midi_hash_sequences'))

    # Run match_one_midi for each MIDI data and MSD index list
    results = joblib.Parallel(n_jobs=11, verbose=51)(
        joblib.delayed(experiment_utils.match_sequence)(
            midi_datas[md5], msd_data, midi_index_mapping[md5], GULLY, PENALTY)
        for md5 in midi_datas)

    # Create DHS match results output path if it doesn't exist
    output_path = os.path.join(RESULTS_PATH, 'dhs_match_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Save list of all matching results
    results_file = os.path.join(output_path, '{}_results.h5'.format(SPLIT))
    deepdish.io.save(results_file, results)
