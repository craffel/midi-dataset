'''
Hash the echonest features for every MSD entry using a pre-made model
'''
import sys
sys.path.append('../')
import hashing_utils
import hash_match
import glob
import cPickle as pickle
import beat_aligned_feats
import hdf5_getters
import os
import numpy as np
import theano
# import joblib
import time
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, n_bits, dropout)

BASE_DATA_PATH = '../data'
output_path = os.path.join(BASE_DATA_PATH, 'msd', 'pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

layers = hashing_utils.build_network(
    (None, 2, 100, 12), num_filters['Y'], filter_size['Y'], ds['Y'],
    hidden_layer_sizes['Y'], dropout, n_bits)
hashing_utils.load_model(layers, '../results/model_Y.pkl')
hash = theano.function(
    [layers[0].input_var], layers[-1].get_output(deterministic=True))

h5_glob = os.path.join(BASE_DATA_PATH, 'msd', 'h5', '*', '*', '*', '*.h5')


def process_one_file(h5_file):
    '''
    Hash the features in a single h5 file.

    :parameters:
        - h5_file : str
            Path to h5 file to process
    '''
    try:
        output_filename = h5_file.replace('h5', 'pkl')
        if os.path.exists(output_filename):
            return
        with hdf5_getters.open_h5_file_read(h5_file) as h5:
            # Load in beat times from MSD
            beats = hdf5_getters.get_beats_start(h5)
            artist = str(hdf5_getters.get_artist_name(h5))
            title = str(hdf5_getters.get_title(h5))
            duration = hdf5_getters.get_duration(h5)
            # Some files have no EN analysis
            if beats.size == 0:
                return
            # and beat-synchronous feature matrices
            chroma = beat_aligned_feats.get_btchromas_loudness(h5).T
            timbre = beat_aligned_feats.get_bttimbre(h5).T
        mean_chroma = chroma.mean(axis=0)
        msd_features = np.array([chroma, timbre])
        if msd_features.shape[1] < 5:
            return
        mean, std = hashing_utils.standardize(msd_features)
        msd_features = (msd_features - mean)/std
        if np.isnan(msd_features).any():
            return
        hashed_features = hash(
            msd_features[np.newaxis].astype(theano.config.floatX))
        hashes = hash_match.vectors_to_ints(hashed_features > 0)
        hashes = hashes.astype('uint16')
        h5_dict = {'artist': artist, 'title': title, 'hash_list': hashes,
                   'duration': duration, 'mean_chroma': mean_chroma}
        output_filename = h5_file.replace('h5', 'pkl')
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        with open(output_filename, 'wb') as f:
            pickle.dump(h5_dict, f)
    except Exception as e:
        print "Error creating {}: {}".format(h5_file, e)
        return

if __name__ == '__main__':
    now = time.time()

    for n, h5_file in enumerate(glob.glob(h5_glob)):
        process_one_file(h5_file)
        if not n % 1000:
            print "{:<7} in {:.3f}".format(n, time.time() - now)
            now = time.time()
'''
joblib.Parallel(n_jobs=10, verbose=1)(joblib.delayed(process_one_file)(h5_file)
                                    for h5_file in glob.glob(h5_glob))
'''
