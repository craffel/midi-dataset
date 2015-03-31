'''
Match entries in the clean MIDI subset to the Million Song Dataset
'''
import sys
sys.path.append('..')
import glob
import cPickle as pickle
import hash_match
import numpy as np
import os
import json
import joblib

# Path to the list of MIDI<->pairs file
PAIR_FILE = '../file_lists/dev_pairs.csv'
# Where to output the matching results
RESULTS_FILE = '../results/dev_results.js'

LENGTH_TOLERANCE = .4
CHROMA_PERCENTILE = 25
GULLY = .9
PENALTY = 4


def path_to_id(pkl_file):
    ''' Convert an MSD pkl file path to the MSD ID '''
    return os.path.splitext(os.path.basename(pkl_file))[0]


# Load in all hashed MSD sequences
data = []
for pkl_file in glob.glob('../data/msd/pkl/*/*/*/*.pkl'):
    with open(pkl_file) as f:
        try:
            data.append(pickle.load(f))
            data[-1]['id'] = path_to_id(pkl_file)
        except Exception as e:
            print "Error loading {}: {}".format(pkl_file, e)

# Extract mean chroma vectors
mean_chromas = np.array([d['mean_chroma'] for d in data])
# Create a separate list of the sequences
sequences = [d['hash_list'] for d in data]
# And a separate list of IDs
msd_ids = [d['id'] for d in data]


# Load in the clean MIDI index
with open('../data/clean_midi/index.js') as f:
    midi_index = json.load(f)


def match_one_midi(midi_data, msd_match_indices):
    '''
    Match a MIDI sequence against the MSD and evaluate whether a good match was
    found

    :parameters:
        - midi_data : dict
            Dict of MIDI data, including mean chroma vector, hash sequence, etc
        - msd_match_indices : list-like of int
            indices of entries in the MSD this MIDI should potentially match to

    :returns:
        - results : dict
            Dictionary with diagnostics about whether this match was successful
    '''
    # Find indices of MSD sequences whose length are within the tolerance
    valid_length_indices = hash_match.filter_by_length(
        midi_data['hash_list'], sequences, LENGTH_TOLERANCE)
    # Find indices of MSD sequences where the distance between MSD/MIDI mean
    # chroma vectors is within the provided percentile
    valid_chroma_indices = hash_match.filter_by_mean_chroma(
        midi_data['mean_chroma'], mean_chromas, CHROMA_PERCENTILE)
    # Intersect these two index sets to determine which indices to use
    valid_indices = np.intersect1d(valid_length_indices, valid_chroma_indices)
    # Match this MIDI sequence against MSD sequences
    matches, scores = hash_match.match_one_sequence(
        midi_data['hash_list'], sequences, GULLY, PENALTY, valid_indices)
    # Store results of the match
    results = {}
    results['midi_md5'] = midi_data['md5']
    results['msd_match_ids'] = [data[n]['id'] for n in msd_match_indices]
    results['msd_match_in_length'] = [(idx in valid_length_indices)
                                      for idx in msd_match_indices]
    results['msd_match_in_chroma'] = [(idx in valid_chroma_indices)
                                      for idx in msd_match_indices]
    # Compile the rank and score for each MSD entry which should match the MIDI
    matched_ranks = []
    matched_scores = []
    for msd_index in msd_match_indices:
        # If it was pruned by mean chroma/length pruning, compute manually
        if msd_index not in matches:
            _, score = hash_match.match_one_sequence(
                midi_data['hash_list'], [sequences[msd_index]], GULLY, PENALTY)
            score = score[0]
            rank = (scores < score).sum()
        # Otherwise find the result
        else:
            rank = matches.index(msd_index)
            score = scores[rank]
        matched_ranks.append(rank)
        matched_scores.append(score)
    results['msd_match_ranks'] = matched_ranks
    results['msd_match_scores'] = matched_scores

    return results

midi_datas = []
msd_match_index_lists = []
with open(PAIR_FILE) as f:
    midi_h5_mapping = [line.strip().split(',') for line in f.readlines()]

for midi_md5, msd_id in midi_h5_mapping:
    midi_entry = [entry for entry in midi_index if entry['md5'] == midi_md5]
    if len(midi_entry) == 0:
        continue
    else:
        midi_entry = midi_entry[0]
    midi_pkl = os.path.join('../data/clean_midi/pkl/',
                            midi_entry['path'] + '.pkl')
    try:
        with open(midi_pkl) as f:
            midi_data = pickle.load(f)
    except:
        continue
    if any([(midi_data['md5'] == d['md5']) for d in midi_datas]):
        continue
    if msd_id not in msd_ids:
        continue
    midi_datas.append(midi_data)
    msd_match_ids = [id for md5, id in midi_h5_mapping
                     if md5 == midi_data['md5']]
    msd_match_indices = []
    for id in msd_match_ids:
        if id in msd_ids:
            msd_match_indices.append(msd_ids.index(id))
    msd_match_index_lists.append(msd_match_indices)

results = joblib.Parallel(n_jobs=11, verbose=10)(
    joblib.delayed(match_one_midi)(midi_data, msd_match_indices)
    for midi_data, msd_match_indices in zip(midi_datas, msd_match_index_lists))

with open(RESULTS_FILE, 'wb') as f:
    json.dump(results, f)
