'''
Match entries in the clean MIDI subset to the Million Song Dataset
'''
import sys
sys.path.append('..')
import glob
import cPickle as pickle
import hash_match
import os
import json
import joblib
import numpy as np

# Path to the list of MIDI<->pairs file
PAIR_FILE = '../file_lists/dev_pairs.csv'
# Where to output the matching results
RESULTS_FILE = '../results/dev_results_alignment_verified.js'
# Where the aligned clean MIDI diagnostics files live
ALIGNMENT_DIAGNOSTICS_PATH = '../data/clean_midi_aligned/npz/'

GULLY = .9
PENALTY = 4
# Ignore all hash sequences below this length (they cause issues)
MIN_SEQUENCE_LENGTH = 30
# A DP score above this means the alignment is bad
SCORE_THRESHOLD = .78


def path_to_id(pkl_file):
    ''' Convert an MSD pkl file path to the MSD ID '''
    return os.path.splitext(os.path.basename(pkl_file))[0]


# Load in all hashed MSD sequences
data = []
for pkl_file in glob.glob('../data/msd/pkl/*/*/*/*.pkl'):
    with open(pkl_file) as f:
        try:
            pkl = pickle.load(f)
            if len(pkl['hash_list']) < MIN_SEQUENCE_LENGTH:
                continue
            pkl['id'] = path_to_id(pkl_file)
            data.append(pkl)
        except Exception as e:
            print "Error loading {}: {}".format(pkl_file, e)

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
    # Match this MIDI sequence against MSD sequences
    matches, scores, n_pruned_dist = hash_match.match_one_sequence(
        midi_data['hash_list'], sequences, GULLY, PENALTY)
    # Store results of the match
    results = {}
    results['midi_md5'] = midi_data['md5']
    results['msd_match_ids'] = [data[n]['id'] for n in msd_match_indices]
    # Compile the rank and score for each MSD entry which should match the MIDI
    matched_ranks = []
    matched_scores = []
    for msd_index in msd_match_indices:
        # If it was pruned by mean chroma/length pruning, compute manually
        if msd_index not in matches:
            _, score, _ = hash_match.match_one_sequence(
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
    results['n_pruned_dist'] = n_pruned_dist

    return results

midi_datas = []
msd_match_index_lists = []
midi_h5_mapping = []
with open(PAIR_FILE) as f:
    for line in f.readlines():
        midi_md5, dataset, msd_id = line.strip().split(',')
        # The pairs.csv files will include pairs from all datasets
        # Only grab those for the MSD
        if dataset == 'msd':
            # Only include if the alignment was successful
            alignment_file = os.path.join(
                ALIGNMENT_DIAGNOSTICS_PATH,
                'msd_{}_{}.npz'.format(msd_id, midi_md5))
            if os.path.exists(alignment_file):
                diagnostics = np.load(alignment_file)
                if diagnostics['score'] < SCORE_THRESHOLD:
                    midi_h5_mapping.append([midi_md5, msd_id])

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

results = joblib.Parallel(n_jobs=11, verbose=51)(
    joblib.delayed(match_one_midi)(midi_data, msd_match_indices)
    for midi_data, msd_match_indices in zip(midi_datas, msd_match_index_lists))

with open(RESULTS_FILE, 'wb') as f:
    json.dump(results, f)
