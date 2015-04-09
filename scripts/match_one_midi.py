'''
Match one MIDI file against MSD hashes
'''
import sys
sys.path.append('..')
import glob
import cPickle as pickle
import pretty_midi
import hashing_utils
import theano
import hash_match
import numpy as np
import time
import argparse
import unicodedata
import tabulate
from network_structure import (hidden_layer_sizes, num_filters, filter_size,
                               ds, n_bits, dropout)

print "Loading in MSD data ..."
# Load in all hashed MSD sequences
data = []
for pkl_file in glob.glob('../data/msd/pkl/*/*/*/*.pkl'):
    with open(pkl_file) as f:
        try:
            data.append(pickle.load(f))
        except:
            pass

# Create a separate list of the sequences
sequences = [d['hash_list'] for d in data]
# And mean chroma vectors
mean_chromas = np.array([d['mean_chroma'] for d in data])

print "Loading in hasher ..."
layers = hashing_utils.build_network(
    (None, 1, 100, 48), num_filters['X'], filter_size['X'], ds['X'],
    hidden_layer_sizes['X'], dropout, n_bits)
hashing_utils.load_model(layers, '../results/model_X.pkl')
hash = theano.function(
    [layers[0].input_var], layers[-1].get_output(deterministic=True))

# Load in training set statistics for standardization
with open('../results/X_mean_std.pkl') as f:
    train_stats = pickle.load(f)


def match_one_midi(midi_file):
    '''
    Hash and match a single MIDI file against the MSD

    :parameters:
        - midi_file : str
            Path to a MIDI file to match
    '''
    # Get a beat-synchronous piano roll of the MIDI
    pm = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = pm.get_piano_roll(times=pm.get_beats()).T
    piano_roll = piano_roll[np.newaxis, :, 36:84]
    # Make the piano roll look like it does when we trained the hasher
    piano_roll = (piano_roll - train_stats['mean'])/train_stats['std']
    hashed_piano_roll = hash(
        piano_roll[np.newaxis].astype(theano.config.floatX))
    # Compute hash sequence
    query = hash_match.vectors_to_ints(hashed_piano_roll > 0)
    query = query.astype('uint16')
    # Get indices of sequences which are within 40% -> 1/40% of this seq length
    valid_length_indices = hash_match.filter_by_length(query, sequences, .4)
    # Compute MIDI mean chroma vector
    query_chroma = pm.get_chroma().mean(axis=1)
    # Get sequences less than the mean chroma distance
    valid_chroma_indices = hash_match.filter_by_mean_chroma(
        query_chroma, mean_chromas, 20)
    # Intersect to get valid index set
    valid_indices = np.intersect1d(valid_length_indices, valid_chroma_indices)
    # Match the MIDI file query hash list against all sequences
    matches, scores = hash_match.match_one_sequence(
        query, sequences, .9, 4, valid_indices)
    return matches, scores

clean = lambda string : unicodedata.normalize(
    'NFKD', unicode(string, 'utf-8', 'ignore')).encode('ascii', 'ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match a single MIDI file')
    parser.add_argument('midi_file', type=str, help='Path to a MIDI file')
    parser.add_argument('artist', nargs='?', type=str, default=None,
                        help='Artist (optional)')
    parser.add_argument('title', nargs='?', type=str, default=None,
                        help='Title (optional)')
    args = vars(parser.parse_args())
    print "Matching ..."
    now = time.time()
    matches, scores = match_one_midi(args['midi_file'])
    print "  took {:.3f}s".format(time.time() - now)
    print "Top 20 matches:"
    print tabulate.tabulate(
        [[n, clean(data[n]['artist']), clean(data[n]['title']), score]
         for n, score in zip(matches[:20], scores[:20])],
        headers=['ID', 'Artist', 'Title', 'Score'])

    if args['title'] is not None and args['artist'] is not None:
        print "Supplied file has duration {}".format(
            pretty_midi.PrettyMIDI(args['midi_file']).get_end_time())
        print "Scores for expected matches:"
        artist = clean(args['artist']).lower()
        title = clean(args['title']).lower()
        expected_matches = [n for n, d in enumerate(data) if (
            title in d['title'].lower() and artist in d['artist'].lower())]
        print tabulate.tabulate(
            [[n, clean(data[n]['artist']), clean(data[n]['title']),
              '--' if n not in matches else scores[matches.index(n)],
              data[n]['duration']] for n in expected_matches],
            headers=['ID', 'Artist', 'Title', 'Score', 'Duration'])
