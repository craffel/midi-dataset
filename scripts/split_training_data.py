'''
Split the hashing training data into training, validation, and test sets.
'''

import os
import numpy as np
import json
import itertools
import csv

# Proportions of the dataset to use for each set
TRAIN_VAL_THRESHOLD = .9
DEV_TEST_THRESHOLD = .5
TRAIN_VAL_OR_DEV_TEST = .5

# Directory to write out results
RESULTS_PATH = os.path.join('..', 'results')

# Get list of all MIDI->mp3 pairs
with open(os.path.join(RESULTS_PATH, 'text_matches.js')) as f:
    pairs = json.load(f)

# Lists of training, validation, and test set files to be populated below
train_files = []
valid_files = []
dev_files = []
test_files = []

# Each pair corresponds to all MIDIs of a song matched to all mp3s of the song
for pair in pairs:
    # Draw a random number to choose train/val or dev/test
    if np.random.rand() < TRAIN_VAL_OR_DEV_TEST:
        # Draw a random number to decide train or val
        if np.random.rand() < TRAIN_VAL_THRESHOLD:
            file_set = train_files
        else:
            file_set = valid_files
    else:
        # Same for dev/test
        if np.random.rand() < DEV_TEST_THRESHOLD:
            file_set = dev_files
        else:
            file_set = test_files
    # Add in all MIDI/mp3 combinations to the chosen file list
    file_set += [[midi_md5, dataset, id]
                 for midi_md5, (dataset, id) in itertools.product(*pair)]


# Write out
def write_list(filename, l):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(l)

write_list(os.path.join(RESULTS_PATH, 'train_pairs.csv'), train_files)
write_list(os.path.join(RESULTS_PATH, 'valid_pairs.csv'), valid_files)
write_list(os.path.join(RESULTS_PATH, 'dev_pairs.csv'), dev_files)
write_list(os.path.join(RESULTS_PATH, 'test_pairs.csv'), test_files)
