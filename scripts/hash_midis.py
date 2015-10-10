'''
Hash the clean_midi dataset using a pre-made model
'''
import sys
sys.path.append('../')
import hashing_utils
import json
import cPickle as pickle
import os
import theano
# import joblib
import time
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, n_bits, dropout)
import lasagne
from hash_msd import process_one_file

BASE_DATA_PATH = '../data'
output_path = os.path.join(BASE_DATA_PATH, 'clean_midi', 'pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

layers = hashing_utils.build_network(
    (None, 1, None, 48), num_filters['X'], filter_size['X'], ds['X'],
    hidden_layer_sizes['X'], dropout, n_bits)
hashing_utils.load_model(layers, '../results/model_X.pkl')
hash = theano.function(
    [layers[0].input_var],
    lasagne.layers.get_output(layers[-1], deterministic=True))

# Load in training set statistics for standardization
with open('../results/X_mean_std.pkl') as f:
    train_stats = pickle.load(f)


if __name__ == '__main__':
    now = time.time()

    with open('../data/clean_midi/index.js') as f:
        midi_index = json.load(f)

    for n, midi_entry in enumerate(midi_index):
        process_one_file(midi_entry, 'clean_midi', train_stats, hash)
        if not n % 1000:
            print "{:<7} in {:.3f}".format(n, time.time() - now)
            now = time.time()
