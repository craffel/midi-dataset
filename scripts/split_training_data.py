'''
Split the hashing training data into training, validation, and test sets.
'''

import os
import numpy as np
import json
import itertools
import csv

# Proportions of the dataset to use for each set
# Train = 60%, validation = 5%, development = 10%, test = 25%
TRAIN_PROB = .6
VALID_PROB = .05
DEV_PROB = .1
TEST_PROB = .25

# Directory to write out results
RESULTS_PATH = os.path.join('..', 'results')

# Get list of all MIDI->mp3 pairs
with open(os.path.join(RESULTS_PATH, 'text_matches.js')) as f:
    pairs = json.load(f)

# List of output files to write
output_files = [
    'train_pairs.csv', 'validate_pairs.csv', 'dev_pairs.csv', 'test_pairs.csv']
# Lists of training, validation, and test set files to be populated below
file_sets = [[] for _ in range(len(output_files))]

# Each pair corresponds to all MIDIs of a song matched to all mp3s of the song
for pair in pairs:
    # Choose a file set to add to according to their desired proportions
    file_set = file_sets[np.random.choice(
        len(file_sets), p=[TRAIN_PROB, VALID_PROB, DEV_PROB, TEST_PROB])]
    # Add in all MIDI/mp3 combinations to the chosen file list
    file_set += [[midi_md5, dataset, id]
                 for midi_md5, (dataset, id) in itertools.product(*pair)]


# Write out
def write_list(filename, l):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(l)

for file_set, filename in zip(file_sets, output_files):
    write_list(os.path.join(RESULTS_PATH, filename), file_set)
