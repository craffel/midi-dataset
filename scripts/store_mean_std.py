'''
Store the mean/std of the training set for later use.
'''

import sys
sys.path.append('../')
import hashing_utils
import os
import numpy as np
import theano
import pickle
import glob

BASE_DATA_PATH = '../data'
X = []
Y = []
# Load in all files
for filename in glob.glob(os.path.join(BASE_DATA_PATH, 'hash_dataset',
                                       'train', 'npz', '*.npz')):
    data = np.load(filename)
    X.append(np.array(
        data['X'], dtype=theano.config.floatX, order='C'))
    Y.append(np.array(
        data['Y'], dtype=theano.config.floatX, order='C'))
X_stats = {}
Y_stats = {}
X_stats['mean'], X_stats['std'] = hashing_utils.standardize(
    np.concatenate(X, axis=1))
Y_stats['mean'], Y_stats['std'] = hashing_utils.standardize(
    np.concatenate(Y, axis=1))

if not os.path.exists('../results'):
    os.makedirs('../results')

with open('../results/X_mean_std.pkl', 'wb') as f:
    pickle.dump(X_stats, f)
with open('../results/Y_mean_std.pkl', 'wb') as f:
    pickle.dump(Y_stats, f)
