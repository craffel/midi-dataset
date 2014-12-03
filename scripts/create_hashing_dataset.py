'''
Given alignment results, create a dataset for hashing using those alignments
which were sucessful (based on simple thresholding of dynamic programming
score) consisting of beatwise MSD features and aligned beatwise MIDI piano roll
'''

import sys
sys.path.append('..')
import numpy as np
import os
import pretty_midi
import hdf5_getters
import beat_aligned_feats
import glob

# A DP score above this means the alignment is bad
SCORE_THRESHOLD = .05
BASE_DATA_PATH = '../data'
DATASETS = ['uspop2002', 'cal10k', 'cal500']

aligned_path = os.path.join(BASE_DATA_PATH, 'clean_midi_aligned')
output_path = os.path.join(BASE_DATA_PATH, 'hash_dataset')

if not os.path.exists(os.path.join(output_path, 'npz')):
    os.makedirs(os.path.join(output_path, 'npz'))

for diagnostics_file in glob.glob(os.path.join(aligned_path, 'npz', '*.npz')):
    diagnostics = np.load(diagnostics_file)
    # Skip bad alignments
    if diagnostics['score'] > SCORE_THRESHOLD:
        continue
    h5_file = str(diagnostics['audio_filename']).replace('mp3', 'h5')
    if not os.path.exists(h5_file):
        continue
    with hdf5_getters.open_h5_file_read(h5_file) as h5:
        # Load in beat times from MSD
        beats = hdf5_getters.get_beats_start(h5)
        # Some files have no EN analysis
        if beats.size == 0:
            continue
        # and beat-synchronous feature matrices
        chroma = beat_aligned_feats.get_btchromas(h5)
        timbre = beat_aligned_feats.get_bttimbre(h5)
        loudness = beat_aligned_feats.get_btloudnessmax(h5)
    # Load in pretty midi object
    pm = pretty_midi.PrettyMIDI(str(diagnostics['midi_filename']))
    start_time = min([n.start for i in pm.instruments for n in i.notes])
    end_time = min(pm.get_end_time(), beats.max())
    # Get indices which fall within the range of correct alignment
    time_mask = np.logical_and(beats >= start_time, beats <= end_time)
    beats = beats[time_mask]
    # Stack it
    msd_features = np.vstack([chroma, timbre, loudness])
    if np.isnan(msd_features).any():
        continue
    # Construct piano roll, aligned to the msd beat times
    piano_roll = pm.get_piano_roll(times=beats)
    # Write out
    np.savez_compressed(os.path.join(output_path, 'npz',
                                     os.path.basename(diagnostics_file)),
                        X=piano_roll, Y=msd_features)
