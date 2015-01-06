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
import theano.tensor as T
import hash_match
import numpy as np
import time

# Load in all hashed MSD sequences
data = []
for pkl_file in glob.glob('../data/msd/pkl/*/*/*.pkl'):
    with open(pkl_file) as f:
        try:
            data.append(pickle.load(f))
        except:
            print 'Error loading {}'.format(pkl_file)

# Store sequence lengths
lengths = np.array([d['hash_list'].shape[0] for d in data])
# Sort data by sequence length
sorted_idx = np.argsort(lengths)
data = [data[n] for n in sorted_idx]
# Create a separate list of the sequences
sequences = [d['hash_list'] for d in data]
lengths = lengths[sorted_idx]

# Load in the MIDI hasher
with open('../results/model_X.pkl') as f:
    hasher_params = pickle.load(f)
hasher_layers = hashing_utils.load_model(hasher_params, 100)
# Create a function for hashing a piano roll
hasher_input = T.matrix('hasher_input')
hash = theano.function([hasher_input], hasher_layers[-1].get_output(
    hasher_input, deterministic=True))


def match_one_midi(midi_file):
    '''
    Hash and match a single MIDI file against the MSD

    :parameters:
        - midi_file : str
            Path to a MIDI file to match
    '''
    # Get a beat-synchronous piano roll of the MIDI
    pm = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = pm.get_piano_roll(times=pm.get_beats())[36:84, :].T
    # Make the piano roll look like it does when we trained the hasher
    piano_roll = hashing_utils.shingle(piano_roll, 4)
    mean, std = hashing_utils.standardize(piano_roll)
    piano_roll = (piano_roll - mean)/std
    hashed_piano_roll = hash(piano_roll.astype(theano.config.floatX))
    # Compute hash sequence
    query = hash_match.vectors_to_ints(hashed_piano_roll > 0)
    query = query.astype('uint16')
    # Match the MIDI file query hash list against all sequences
    matches, scores = hash_match.match_one_sequence(
        query, sequences, lengths, .1, 25, .95, 8)
    return matches, scores


if __name__ == '__main__':
    # Get input MIDI filename
    midi_filename = sys.argv[1]
    now = time.time()
    matches, scores = match_one_midi(midi_filename)
    print "Matching took {}s".format(time.time() - now)
    print "Top 20 matches:"
    for n, score in zip(matches[:20], scores[:20]):
        print '{} - {} : {}'.format(data[n]['artist'], data[n]['title'], score)
