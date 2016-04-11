"""
Precompute statistics-based embeddings (stacked trackwise mean and standard
deviation of feature dimensions) for all MSD entries and MIDI files from
clean_midi
"""
# We'll use msgpack for I/O.  It seems to be fastest, and is widely supported.
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import deepdish
import os
import numpy as np
import traceback
import sys
sys.path.append(os.path.join('..', '..'))
import whoosh_search

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'


def compute_statistics(gram):
    '''
    Computes the mean and standard deviation of feature dimensions in a
    provided feature vector sequence.

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram, shape=(n_frames, n_frequency_bins)

    Returns
    -------
    statistics : np.ndarray
        Stacked mean and standard deviation.
    '''
    # Compute and stack statistics, adding a "n_samples" dimension in front
    return np.concatenate((gram.mean(axis=0), gram.std(axis=0)))[np.newaxis]

if __name__ == '__main__':
    # Compute mean/std embeddings for each dataset/modality
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
        # Load in CQTs and write out downsampled hash sequences
        for entry in file_list:
            try:
                # Construct CQT h5 file path from file index entry
                h5_file = os.path.join(
                    DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                # Load in CQT
                gram = deepdish.io.load(h5_file)['gram']
                # Compute embedding for this sequence
                embedding = compute_statistics(gram)
                # Construct output path to the same location in
                # RESULTS_PATH/stats_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH, 'stats_{}_embeddings'.format(dataset),
                    entry['path'] + '.mpk')
                # Construct intermediate subdirectories if they don't exist
                if not os.path.exists(os.path.split(output_file)[0]):
                    os.makedirs(os.path.split(output_file)[0])
                # Save result, along with the index entry for convenience
                with open(output_file, 'wb') as f:
                    f.write(msgpack.packb(
                        dict(embedding=embedding, **entry)))
            except Exception:
                print "Error processing : {}".format(h5_file)
                print traceback.format_exc()
