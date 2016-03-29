"""
Precompute thresholded piecewise aggregate approximation sequences for all MSD
entries and MIDI files from clean_midi
"""
# We'll use msgpack for I/O.  It seems to be fastest, and is widely supported.
import msgpack
import msgpack_numpy
import deepdish
import glob
import numpy as np
import dhs
import traceback
import os
import sys
import collections
sys.path.append(os.path.join('..', '..'))
import whoosh_search
msgpack_numpy.patch()

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'


def paa(sequence, window):
    """
    Compute ``Piecewise Aggregate Approximation'' of a sequence, which simply
    computes the local mean of the sequence over non-overlapping windows

    Parameters
    ----------
    sequence : np.ndarray
        A sequence; will be aggregated over its first dimension.
    window : int
        Size of windows over which to compute the mean

    Returns
    -------
    aggregated_sequence : np.ndarray
        Aggregated sequence, the first dimension will be 1/window of its
        original size
    """
    # Truncate sequence to rounded-down nearest divisor
    sequence = sequence[:window*int(sequence.shape[0]/window)]
    # Reshape sequence to be of shape
    # (original_shape[0]/window, window, original trailing dimensions...)
    sequence = sequence.reshape(
            sequence.shape[0]/window, window, sequence.shape[-1])
    # Compute mean over the window dimension
    return sequence.mean(axis=1)

if __name__ == '__main__':
    # Load in training data for computing mean for thresholding
    training_data = collections.defaultdict(list)
    for f in glob.glob(os.path.join(
            RESULTS_PATH, 'training_dataset', 'train', 'h5', '*.h5')):
        for k, v in deepdish.io.load(f).items():
            training_data[k].append(v)
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
        # Compute the mean of the feature dimensions over the training set for
        # thresholding
        input_mean = np.mean(
            np.concatenate(training_data[network], axis=1), axis=1)
        # Load in CQTs and write out downsampled hash sequences
        for entry in file_list:
            try:
                # Construct CQT h5 file path from file index entry
                h5_file = os.path.join(
                    DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                # Load in CQT
                gram = deepdish.io.load(h5_file)['gram']
                # Compute downsampled sequence
                hash_sequence = paa(gram, 8)
                # Threshold using the training set per-dimension mean
                hash_sequence = hash_sequence > input_mean
                # The hash sequence will have feature dimensionality of 48.
                # We want 32-bit vectors, so we need to remove 16 elements.
                # The least informative are probably those on the top and
                # bottom, so remove those.
                hash_sequence = hash_sequence[:, 8:-8]
                # Convert to sequence of integers
                hash_sequence = dhs.vectors_to_ints(hash_sequence)
                hash_sequence = hash_sequence.astype(np.uint32)
                # Construct output path to the same location in
                # RESULTS_PATH/tpaa_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH, 'tpaa_{}_hash_sequences'.format(dataset),
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
