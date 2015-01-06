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
import theano.tensor as T

BASE_DATA_PATH = '../data'
output_path = os.path.join(BASE_DATA_PATH, 'msd', 'pkl')

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open('../results/model_Y.pkl') as f:
    hasher_params = pickle.load(f)

hasher_layers = hashing_utils.load_model(hasher_params, 100)
hasher_input = T.matrix('hasher_input')
hash = theano.function([hasher_input],
                       hasher_layers[-1].get_output(hasher_input,
                                                    deterministic=True))

h5_glob = os.path.join(BASE_DATA_PATH, 'msd', 'h5', '*', '*', '*.h5')
for h5_file in glob.glob(h5_glob):
    with hdf5_getters.open_h5_file_read(h5_file) as h5:
        # Load in beat times from MSD
        beats = hdf5_getters.get_beats_start(h5)
        artist = str(hdf5_getters.get_artist_name(h5))
        title = str(hdf5_getters.get_title(h5))
        duration = hdf5_getters.get_duration(h5)
        # Some files have no EN analysis
        if beats.size == 0:
            continue
        # and beat-synchronous feature matrices
        chroma = beat_aligned_feats.get_btchromas(h5)
        timbre = beat_aligned_feats.get_bttimbre(h5)
        loudness = beat_aligned_feats.get_btloudnessmax(h5)
    msd_features = np.vstack([chroma, timbre, loudness])
    msd_features = msd_features.T
    msd_features = hashing_utils.shingle(msd_features, 4)
    mean, std = hashing_utils.standardize(msd_features)
    msd_features = (msd_features - mean)/std
    if np.isnan(msd_features).any():
        continue
    hashed_features = hash(msd_features.astype(theano.config.floatX))
    hashes = hash_match.vectors_to_ints(hashed_features > 0).astype('uint16')
    h5_dict = {'artist': artist, 'title': title,
               'hash_list': hashes, 'duration': duration}
    output_filename = h5_file.replace('h5', 'pkl')
    if not os.path.exists(os.path.split(output_filename)[0]):
        os.makedirs(os.path.split(output_filename)[0])
    with open(output_filename, 'wb') as f:
        pickle.dump(h5_dict, f)
