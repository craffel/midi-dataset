'''
Creates a .js file of alignment results/metadata, using npz filenames.
Sort of a hack, but it works.
'''

import glob
import sys
sys.path.append('..')
import alignment_analysis
import os
import json
import numpy as np

BASE_DATA_PATH = '../data/'
ALIGNMENT_FOLDER = 'clean_midi_aligned'
DATASETS = ['uspop2002', 'cal10k', 'cal500']

dataset_lists = {}
for dataset in DATASETS:
    with open(os.path.join(BASE_DATA_PATH, dataset, 'index.js')) as f:
        dataset_lists[dataset] = json.load(f)
with open(os.path.join(BASE_DATA_PATH, 'clean_midi', 'index.js')) as f:
    dataset_lists['clean_midi'] = json.load(f)


def process_one_file(diagnostics_file):
    '''
    Get all fields using a given diagnostics file
    '''
    diagnostics_base = os.path.splitext(os.path.split(diagnostics_file)[1])[0]
    dataset, dataset_id, midi_md5 = diagnostics_base.split('_')
    dataset_id = int(dataset_id)
    audio_entry = dataset_lists[dataset][dataset_id]
    audio_name = '{} - {}'.format(audio_entry['artist'], audio_entry['title'])
    midi_entry = [entry for entry in dataset_lists['clean_midi']
                  if entry['md5'] == unicode(midi_md5)]
    assert len(midi_entry) == 1
    midi_entry = midi_entry[0]
    midi_name = '{} - {}'.format(midi_entry['artist'], midi_entry['title'])
    diagnostics = np.load(diagnostics_file)
    scores = alignment_analysis.get_scores(diagnostics['similarity_matrix'],
                                           diagnostics['p'], diagnostics['q'],
                                           float(diagnostics['score']))
    row = [diagnostics_file, dataset, dataset_id, midi_md5, audio_name,
           midi_name] + list(scores) + ['']
    return [str(n) for n in row]

npz_path = os.path.join(BASE_DATA_PATH, ALIGNMENT_FOLDER, 'npz', '*.npz')
tsv_file = os.path.join(BASE_DATA_PATH, ALIGNMENT_FOLDER, 'results.tsv')
with open(tsv_file, 'wb') as f:
    for diagnostics_file in glob.glob(npz_path):
        f.write('\t'.join(process_one_file(diagnostics_file)) + '\n')
