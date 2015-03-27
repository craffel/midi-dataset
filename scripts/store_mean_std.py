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

base_data_directory = '../data'
hash_data_directory = os.path.join(base_data_directory, 'hash_dataset')
with open(os.path.join(hash_data_directory, 'train.csv')) as f:
    train_list = f.read().splitlines()
X = []
Y = []
# Load in all files
for filename in train_list:
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

with open('../results/X_mean_std.pkl', 'wb') as f:
    pickle.dump(X_stats, f)
with open('../results/Y_mean_std.pkl', 'wb') as f:
    pickle.dump(Y_stats, f)
