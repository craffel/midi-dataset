'''
Match entries in the MSD to clean MIDIs, avoiding those used for training
'''
import os
import sys
sys.path.append('..')
import whoosh_search
import json
import numpy as np

BASE_DATA_PATH = '../data/'
FILE_LIST_PATH = '../file_lists/'
MIDI_PATH = 'clean_midi'
DATASETS = ['uspop2002', 'cal10k', 'cal500']
DEV_PROPORTION = .2

# Load in the js filelist for the MIDI dataset
midi_js = os.path.join(BASE_DATA_PATH, MIDI_PATH, 'index.js')
with open(midi_js) as f:
    midi_list = json.load(f)

# Construct a list of MIDI-audio matches, which we will avoid
midis_in_datasets = []
for dataset in DATASETS:
    # Load the whoosh index for this dataset
    index_path = os.path.join(BASE_DATA_PATH, dataset, 'index')
    index = whoosh_search.get_whoosh_index(index_path)
    with index.searcher() as searcher:
        for midi_entry in midi_list:
            # Match each MIDI file entry against this dataset
            matches = whoosh_search.search(searcher, index.schema,
                                           midi_entry['artist'],
                                           midi_entry['title'])
            # If this MIDI file matched anything in the dataset, remember it
            if len(matches) > 0:
                midis_in_datasets.append(midi_entry)

# Load the whoosh index for the MSD
index_path = os.path.join(BASE_DATA_PATH, 'msd', 'index')
index = whoosh_search.get_whoosh_index(index_path)
# List of MIDI-MSD matches for the train set
train_set = []
with index.searcher() as searcher:
    for entry in midis_in_datasets:
        # If we've already added MSD entries for this MIDI, skip
        if any(entry['md5'] == md5 for md5, _ in train_set):
            continue
        # Match each MIDI file entry against the MSD
        matches = whoosh_search.search(searcher, index.schema,
                                       entry['artist'],
                                       entry['title'])
        for match in matches:
            train_set.append([entry['md5'], match[0]])
        # Remove these from the MIDI list for test set
        try:
            del midi_list[midi_list.index(entry)]
        except ValueError:
            continue

with open(os.path.join(FILE_LIST_PATH, 'train_pairs.csv'), 'wb') as f:
    f.write('\n'.join(['{},{}'.format(*match) for match in train_set]))

# List of MIDI-MSD matches
dev_set = []
test_set = []
with index.searcher() as searcher:
    for midi_entry in midi_list:
        # Match each MIDI file entry against the MSD
        matches = whoosh_search.search(searcher, index.schema,
                                       midi_entry['artist'],
                                       midi_entry['title'])
        # Randomly choose whether to place in dev or test set
        if np.random.rand() < DEV_PROPORTION:
            set = dev_set
        else:
            set = test_set
        for match in matches:
            set.append([midi_entry['md5'], match[0]])

with open(os.path.join(FILE_LIST_PATH, 'dev_pairs.csv'), 'wb') as f:
    f.write('\n'.join(['{},{}'.format(*match) for match in dev_set]))
with open(os.path.join(FILE_LIST_PATH, 'test_pairs.csv'), 'wb') as f:
    f.write('\n'.join(['{},{}'.format(*match) for match in test_set]))
