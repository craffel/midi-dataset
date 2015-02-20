'''
Split the hashing training data into training, validation, and test sets.
'''

import os
import sys
sys.path.append('..')
import glob
import numpy as np
import json
import whoosh_search

# Proportions of the dataset to use for each set
TRAIN_PROPORTION = .9
VALID_PROPORTION = .1

# Directory to hashing dataset
BASE_DATA_DIRECTORY = '../data'
training_data_directory = os.path.join(
    BASE_DATA_DIRECTORY, 'hash_dataset', 'npz')

# Get list of all npz files
files = list(glob.glob(os.path.join(training_data_directory, '*.npz')))

# Lists of training, validation, and test set files to be populated below
train_files = []
valid_files = []

# Load in metadata lists of all datasets
indices = {}
for dataset in ['clean_midi', 'cal10k', 'cal500', 'uspop2002']:
    with open(os.path.join(BASE_DATA_DIRECTORY, dataset, 'index.js')) as f:
        indices[dataset] = json.load(f)

# Load in whoosh indices
whooshes = {}
for dataset in ['cal10k', 'cal500', 'uspop2002']:
    with open(os.path.join(BASE_DATA_DIRECTORY, dataset, 'index.js')) as f:
        whooshes[dataset] = whoosh_search.get_whoosh_index(
            os.path.join(BASE_DATA_DIRECTORY, dataset, 'index'))

while len(files) > 0:
    # Draw a random number [0 -> 1] to choose which set to place files in
    random = np.random.rand()
    if random <= TRAIN_PROPORTION:
        file_set = train_files
    else:
        file_set = valid_files
    # Pop the last file from the list
    last_file = files[-1]
    del files[-1]
    file_set.append(os.path.abspath(last_file))
    # Get the file basename
    basename = os.path.splitext(os.path.split(last_file)[-1])[0]
    # Get the dataset, ID, and MIDI md5 from the basename
    dataset, audio_id, midi_id = basename.split('_')
    # Retrieve metadata for this MIDI ID
    midi_entry = [e for e in indices['clean_midi'] if e['md5'] == midi_id][0]
    # Get md5s of MIDIs with the same artist/title
    same_ids = [e['md5'] for e in indices['clean_midi']
                if (e['artist'] == midi_entry['artist']
                    and e['title'] == midi_entry['title'])]
    # Find all IDs in all datasets with the same artist/title
    for dataset in ['cal10k', 'cal500', 'uspop2002']:
        with whooshes[dataset].searcher() as searcher:
            matches = whoosh_search.search(
                searcher, whooshes[dataset].schema, midi_entry['artist'],
                midi_entry['title'])
            same_ids += ['{}_{}'.format(dataset, m[0]) for m in matches]
    # Get unique list of files to add to the set
    files_to_add = set([filename for filename in files for id in same_ids
                        if id in filename])
    file_set += [os.path.abspath(file) for file in files_to_add]
    for filename in files_to_add:
        del files[files.index(filename)]


# Write out
def write_list(filename, l):
    with open(filename, 'wb') as f:
        f.writelines([n + '\n' for n in l])

hash_dataset_directory = os.path.join(BASE_DATA_DIRECTORY, 'hash_dataset')
write_list(os.path.join(hash_dataset_directory, 'train.csv'), train_files)
write_list(os.path.join(hash_dataset_directory, 'valid.csv'), valid_files)
