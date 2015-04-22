'''
Collect groups of same-song MIDI files and match them to groups of same-song
audio files
'''
import sys
sys.path.append('..')
import whoosh_search
import json
import os

# The datasets to match MIDIs against
DATASETS = ['cal10k', 'cal500', 'uspop2002', 'msd']
BASE_DATA_PATH = '../data'

# Load in list of MIDI files
with open(os.path.join(BASE_DATA_PATH, 'clean_midi', 'index.js')) as f:
    midi_list = json.load(f)

# Create dict of whoosh indices
indices = {}
for dataset in DATASETS:
    # Load in whoosh index for this dataset
    indices[dataset] = whoosh_search.get_whoosh_index(
        os.path.join(BASE_DATA_PATH, dataset, 'index'))

pairs = []


def merge_entries(indices):
    '''
    Merges entries in the pairs list given their indices
    '''
    # Don't want to merge one thing
    assert len(indices) > 1
    # Sort the indices so we merge later indices to the first one
    indices = sorted(indices)
    first_index = indices[0]
    for index in indices[1:]:
        # Add in midi md5s if they aren't already in the first list
        for midi_md5 in pairs[index][0]:
            if midi_md5 not in pairs[first_index][0]:
                pairs[first_index][0].append(midi_md5)
        # Same for [dataset, id] pairs
        for dataset_entry in pairs[index][1]:
            if dataset_entry not in pairs[first_index][1]:
                pairs[first_index][1].append(dataset_entry)
    # Now, remove the entries we merged
    indices = sorted(indices, reverse=True)
    for index in indices[:-1]:
        del pairs[index]

# We will remove used entries from the MIDI list as we go until there are none
while len(midi_list) > 0:
    # Pop the last entry off the list
    midi_entry = midi_list[-1]
    # Get all entries with the same artist/title
    midi_matches = [e for e in midi_list
                    if (e['artist'] == midi_entry['artist']
                        and e['title'] == midi_entry['title'])]
    # Remove these matches so we don't use them more than once
    for match in midi_matches:
        del midi_list[midi_list.index(match)]
    # This should never happen
    if len(midi_matches) == 0:
        print "Error: No matches found for {}".format(midi_entry)
    # Match each of these MIDIs against each dataset
    dataset_matches = []
    for dataset in DATASETS:
        with indices[dataset].searcher() as searcher:
            matches = whoosh_search.search(
                searcher, indices[dataset].schema, midi_entry['artist'],
                midi_entry['title'])
            # Add the each matched dataset entry in if we haven't already
            for match in matches:
                if [dataset, match[0]] not in dataset_matches:
                    dataset_matches.append([dataset, match[0]])
    # If there are any matches, add them to pairs
    if len(dataset_matches) > 0:
        pairs.append([[m['md5'] for m in midi_matches], dataset_matches])
        # Find other pairs which have include one of these dataset entries
        merge_indices = []
        for n, pair in enumerate(pairs):
            for dataset_match in dataset_matches:
                if dataset_match in pair[1]:
                    merge_indices.append(n)
                    break
        # We should have found at least one!
        assert len(merge_indices) > 0
        # Merge the rest
        if len(merge_indices) > 1:
            merge_entries(merge_indices)

with open('../file_lists/text_matches.js', 'wb') as f:
    json.dump(pairs, f)
