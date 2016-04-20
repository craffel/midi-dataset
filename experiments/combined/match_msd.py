'''
Match entries in the clean MIDI subset to the Million Song Dataset using DHS,
followed by PSE, followed by DTW
'''
import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join('..', '..', 'scripts'))
import match
import whoosh_search
import experiment_utils
import os
import joblib
import deepdish
import numpy as np

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'
# Should we use the test set or development set?
SPLIT = 'dev'
# A DP score above this means the alignment is correct
SCORE_THRESHOLD = .5
# Skip any hash sequences shorter than this
MIN_SEQUENCE_LENGTH = 30


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

    # Get a list of valid MIDI-MSD match pairs
    midi_msd_mapping = experiment_utils.get_valid_matches(
        os.path.join(RESULTS_PATH, '{}_pairs.csv'.format(SPLIT)),
        SCORE_THRESHOLD,
        os.path.join(RESULTS_PATH, 'clean_midi_aligned', 'h5'))

    # Collect a list of valid MIDI entries in the provided mapping
    valid_midi_list = []
    for midi_md5 in midi_msd_mapping:
        midi_entry = [entry for entry in midi_list if entry['id'] == midi_md5]
        # Edge case - no entry in the MIDI list for this md5
        if len(midi_entry) == 0:
            continue
        else:
            valid_midi_list.append(midi_entry[0])

    # Load all pre-computed MIDI CQTs
    midi_grams = [deepdish.io.load(os.path.join(DATA_PATH, 'clean_midi', 'h5',
                                                entry['path'] + '.h5'))['gram']
                  for entry in valid_midi_list]
    # Load in MIDI hash sequences
    midi_sequences = experiment_utils.load_precomputed_data(
        valid_midi_list,
        os.path.join(RESULTS_PATH, 'dhs_clean_midi_hash_sequences'))
    # Create list hash sequences from loaded-in objects
    midi_sequences = [d['hash_sequence'] for d in midi_sequences]
    # Load all MIDI embeddings
    midi_embeddings = experiment_utils.load_precomputed_data(
        valid_midi_list,
        os.path.join(RESULTS_PATH, 'pse_clean_midi_embeddings'))
    # Concatenate into one giant matrix
    midi_embeddings = np.concatenate(
        [d['embedding'] for d in midi_embeddings], axis=0)

    # Run match_one_midi for each MIDI file
    results = joblib.Parallel(n_jobs=11, verbose=51)(
        joblib.delayed(match.match_one_midi)(
            midi_gram, midi_embedding, midi_sequence, msd_embeddings,
            msd_sequences, msd_feature_paths, msd_ids)
        for midi_gram, midi_embedding, midi_sequence
        in zip(midi_grams, midi_embeddings, midi_sequences))

    full_results = []
    for midi_entry, msd_scores in zip(valid_midi_list, results):
        msd_scores.sort(key=lambda x: -x[1])
        sorted_msd_ids = [e[0] for e in msd_scores]
        sorted_scores = [e[1] for e in msd_scores]
        result = {}
        result['midi_md5'] = midi_entry['id']
        result['msd_match_ids'] = midi_msd_mapping[midi_entry['id']]
        result['msd_match_ranks'] = [
            sorted_msd_ids.index(i) if i in sorted_msd_ids else 1000000
            for i in midi_msd_mapping[midi_entry['id']]]
        result['msd_match_scores'] = [
            sorted_scores[sorted_msd_ids.index(i)]
            if i in sorted_msd_ids else 0.
            for i in midi_msd_mapping[midi_entry['id']]]
        full_results.append(result)

    # Create DHS match results output path if it doesn't exist
    output_path = os.path.join(RESULTS_PATH, 'combined_match_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Save list of all matching results
    results_file = os.path.join(output_path, '{}_results.h5'.format(SPLIT))
    deepdish.io.save(results_file, full_results)
