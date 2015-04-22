'''
Hash the echonest features for every MSD entry using a pre-made model
'''
import sys
sys.path.append('../')
import hashing_utils
import hash_match
import json
import cPickle as pickle
import os
import numpy as np
import theano
# import joblib
import time
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, dropout, n_bits)

BASE_DATA_PATH = '../data'
output_path = os.path.join(BASE_DATA_PATH, 'msd', 'pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

layers = hashing_utils.build_network(
    (None, 1, 100, 48), num_filters['Y'], filter_size['Y'], ds['Y'],
    hidden_layer_sizes['Y'], dropout, n_bits)
hashing_utils.load_model(layers, '../results/model_Y.pkl')
hash = theano.function(
    [layers[0].input_var], layers[-1].get_output(deterministic=True))

npz_glob = os.path.join(BASE_DATA_PATH, 'msd', 'npz', '*', '*', '*', '*.npz')

# Load in training set statistics for standardization
with open('../results/Y_mean_std.pkl') as f:
    train_stats = pickle.load(f)


def process_one_file(index_entry, base_path='msd', train_stats=train_stats,
                     hash=hash):
    '''
    Hash the features in a single npz file.

    :parameters:
        - index_entry : dict
            Entry in an index with keys 'path', 'artist', and 'title'
        - base_path : str
            Which dataset are we processing?
        - train_stats : dict
            Dict where train_stats['mean'] is the training set mean feature
            and train_stats['std'] is the per-feature std dev
        - hash : theano.function
            Theano function which takes in feature matrix and outputs hashes
    '''
    try:
        npz_file = os.path.join(BASE_DATA_PATH, base_path, 'npz',
                                index_entry['path'] + '.npz')
        output_filename = npz_file.replace('npz', 'pkl')
        if os.path.exists(output_filename):
            return
        features = np.load(npz_file)
        sync_gram = features['sync_gram']
        if sync_gram.shape[0] < 6:
            return
        mean_cqt = sync_gram.mean(axis=0)
        sync_gram = sync_gram[np.newaxis]
        sync_gram = (sync_gram - train_stats['mean'])/train_stats['std']
        if np.isnan(sync_gram).any():
            return
        hashed_features = hash(
            sync_gram[np.newaxis].astype(theano.config.floatX))
        hashes = hash_match.vectors_to_ints(hashed_features > 0)
        hashes = hashes.astype('uint16')
        output_dict = dict([('hash_list', hashes), ('mean_cqt', mean_cqt)],
                           **index_entry)
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        with open(output_filename, 'wb') as f:
            pickle.dump(output_dict, f)
    except Exception as e:
        print "Error creating {}: {}".format(index_entry['path'], e)
        return

if __name__ == '__main__':
    now = time.time()

    with open('../data/msd/index.js') as f:
        msd_index = json.load(f)

    for n, msd_entry in enumerate(msd_index):
        process_one_file(msd_entry)
        if not n % 1000:
            print "{:<7} in {:.3f}".format(n, time.time() - now)
