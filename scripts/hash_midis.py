'''
Hash the clean_midi dataset using a pre-made model
'''
import sys
sys.path.append('../')
import hashing_utils
import hash_match
import json
import cPickle as pickle
import pretty_midi
import os
import numpy as np
import theano
# import joblib
import time
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, n_bits, dropout)

BASE_DATA_PATH = '../data'
output_path = os.path.join(BASE_DATA_PATH, 'clean_midi', 'pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

layers = hashing_utils.build_network(
    (None, 1, 100, 48), num_filters['X'], filter_size['X'], ds['X'],
    hidden_layer_sizes['X'], dropout, n_bits)
hashing_utils.load_model(layers, '../results/model_X.pkl')
hash = theano.function(
    [layers[0].input_var], layers[-1].get_output(deterministic=True))

# Load in training set statistics for standardization
with open('../results/X_mean_std.pkl') as f:
    train_stats = pickle.load(f)


def process_one_file(midi_entry):
    '''
    Hash the piano roll in a single MIDI file

    :parameters:
        - midi_entry : dict
            An entry from the clean_midi file index
    '''
    try:
        midi_file = os.path.join(BASE_DATA_PATH, 'clean_midi', 'mid',
                                 midi_entry['path'] + '.mid')
        output_filename = os.path.join(BASE_DATA_PATH, 'clean_midi', 'pkl',
                                       midi_entry['path'] + '.pkl')
        if os.path.exists(output_filename):
            return
        pm = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = pm.get_piano_roll(times=pm.get_beats()).T
        piano_roll = piano_roll[np.newaxis, :, 36:84]
        # Make the piano roll look like it does when we trained the hasher
        piano_roll = (piano_roll - train_stats['mean'])/train_stats['std']
        hashed_piano_roll = hash(
            piano_roll[np.newaxis].astype(theano.config.floatX))
        piano_roll_sequence = hash_match.vectors_to_ints(hashed_piano_roll > 0)
        piano_roll_sequence = piano_roll_sequence.astype(np.uint16)
        mean_chroma = pm.get_chroma().mean(axis=1)
        midi_dict = {'artist': midi_entry['artist'],
                     'title': midi_entry['title'],
                     'md5': midi_entry['md5'],
                     'mean_chroma': mean_chroma,
                     'hash_list': piano_roll_sequence}
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        with open(output_filename, 'wb') as f:
            pickle.dump(midi_dict, f)
    except Exception as e:
        print "Error creating {}: {}".format(midi_file, e)
        return

if __name__ == '__main__':
    now = time.time()

    with open('../data/clean_midi/index.js') as f:
        midi_index = json.load(f)

    for n, midi_entry in enumerate(midi_index):
        process_one_file(midi_entry)
        if not n % 1000:
            print "{:<7} in {:.3f}".format(n, time.time() - now)
            now = time.time()
