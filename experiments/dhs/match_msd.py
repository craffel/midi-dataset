'''
Match entries in the clean MIDI subset to the Million Song Dataset
'''
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import sys
sys.path.append('../../')
import whoosh_search
import dhs
import os
import joblib
import utils
import deepdish

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'
# Should we use the test set or development set?
SPLIT = 'dev'
# DTW parameters
GULLY = .9
PENALTY = 4
# Ignore all hash sequences below this length (they cause issues)
MIN_SEQUENCE_LENGTH = 30
# A DP score above this means the alignment is bad
SCORE_THRESHOLD = .5


def match_one_midi(midi_data, msd_data, msd_match_indices):
    '''
    Match a MIDI sequence against the MSD and evaluate whether a good match was
    found

    Parameters
    ----------
    midi_data : dict
        Dict of MIDI data, including hash sequence
    sequences : list of dict
        List of MSD entries (hash sequences, metadata) to match against
    msd_match_indices : list-like of int
        Indices of entries in the sequences this MIDI should potentially match

    Returns
    -------
    results : dict
        Dictionary with diagnostics about whether this match was successful
    '''
    # Create a separate list of the sequences
    msd_sequences = [d['hash_sequence'] for d in msd_data]
    # Match this MIDI sequence against MSD sequences
    matches, scores, n_pruned_dist = dhs.match_one_sequence(
        midi_data['hash_sequence'], msd_sequences, GULLY, PENALTY, prune=False)
    # Store results of the match
    results = {}
    results['midi_md5'] = midi_data['id']
    results['msd_match_ids'] = [msd_data[n]['id'] for n in msd_match_indices]
    # Compile the rank and score for each MSD entry which should match the MIDI
    results['msd_match_ranks'] = [
        matches.index(msd_index) for msd_index in msd_match_indices]
    results['msd_match_scores'] = [
        scores[rank] for rank in results['msd_match_ranks']]
    results['n_pruned_dist'] = n_pruned_dist
    return results

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
    msd_data = []
    for entry in msd_list:
        mpk_file = os.path.join(
            RESULTS_PATH, 'dhs_msd_hash_sequences', entry['path'] + '.mpk')
        # If creating a CQT or hashing failed, there will be no file
        if os.path.exists(mpk_file):
            try:
                with open(mpk_file) as f:
                    data = msgpack.unpackb(f.read())
                # Ignore very short sequences
                if len(data['hash_sequence']) < MIN_SEQUENCE_LENGTH:
                    continue
                msd_data.append(data)
            except Exception as e:
                print "Error loading {}: {}".format(mpk_file, e)
    # Create a separate list of the IDs of each entry in msd_data
    msd_data_ids = [d['id'] for d in msd_data]

    # Get a list of valid MIDI-MSD match pairs
    midi_msd_mapping = utils.get_valid_matches(
        os.path.join(RESULTS_PATH, '{}_pairs.csv'.format(SPLIT)),
        SCORE_THRESHOLD,
        os.path.join(RESULTS_PATH, 'clean_midi_aligned', 'h5'))

    # We will create a new dict which only contains indices of correct matches
    # in the msd_sequences list, and only for matches we could load in
    midi_index_mapping = {}
    # Also create dict of loaded MIDI data
    midi_datas = {}
    for midi_md5, msd_ids in midi_msd_mapping.items():
        midi_entry = [entry for entry in midi_list if entry['id'] == midi_md5]
        # Edge case - no entry in the MIDI list for this md5
        if len(midi_entry) == 0:
            continue
        else:
            midi_entry = midi_entry[0]
        # Load in data (hash sequence and metadata) for this MIDI file
        midi_mpk = os.path.join(RESULTS_PATH, 'dhs_clean_midi_hash_sequences',
                                midi_entry['path'] + '.mpk')
        try:
            with open(midi_mpk) as f:
                midi_data = msgpack.unpackb(f.read())
        except Exception as e:
            continue
        # We were successful, so save this data and matching indices
        midi_datas[midi_md5] = midi_data
        midi_index_mapping[midi_md5] = [msd_data_ids.index(i) for i in msd_ids]

    # Run match_one_midi for each MIDI data and MSD index list
    results = joblib.Parallel(n_jobs=11, verbose=51)(
        joblib.delayed(match_one_midi)(
            midi_datas[md5], msd_data, midi_index_mapping[md5])
        for md5 in midi_datas)

    # Create DHS match results output path if it doesn't exist
    output_path = os.path.join(RESULTS_PATH, 'dhs_match_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Save list of all matching results
    results_file = os.path.join(output_path, '{}_results.h5'.format(SPLIT))
    deepdish.io.save(results_file, results)
