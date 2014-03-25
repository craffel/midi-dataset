# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Functions for creating the dataset to be used for cross-modality hashing experiments.
'''

# <codecell>

import numpy as np
import csv
import os
import midi
import pretty_midi
import hdf5_getters
import beat_aligned_feats
import glob

# <codecell>

def hms_to_s(hms):
    ''' Given an hour:minute:second string, resturn seconds '''
    h, m, s = hms.split(':')
    return 60*60*int(h) + 60*int(m) + int(s)

# <codecell>

def load_results(path_to_tsv):
    ''' 
    Given a tab-separated value file with entries in the format
        Bryan Adams - Summer Of '69.mp3	70.3806	0:00:03	0:03:12	Some egregious pitch bends
    return a list of the files which have start/end times listed and the start/end times.
    '''
    files = []
    start_times = []
    end_times = []
    with open(path_to_tsv) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            if len(line) == 5 and len(line[2].split(':')) == 3 and len(line[3].split(':')) == 3:
                files.append(line[0])
                start_times.append(hms_to_s(line[2]))
                end_times.append(hms_to_s(line[3]))
    return files, np.array(start_times), np.array(end_times)

# <codecell>

def get_data_folder(filename):
    ''' Given an mp3 filename, returns cal500 or cal10k depending on where the file is (a hack) '''
    if ' ' in filename:
        return 'cal10k'
    else:
        return 'cal500'

# <codecell>

# Create the hashing dataset based on which files are aligned, and when they are
if __name__=='__main__':
    # Set up paths
    base_data_path = 'data'
    aligned_path = os.path.join(base_data_path, 'aligned')
    tsv_path = os.path.join(aligned_path, 'results.tsv')
    output_path = os.path.join(base_data_path, 'hash_dataset')
    midi_directory = 'midi-aligned-new-new-dpmod-multiple-files'
    
    def to_numbered_mid(filename):
        ''' Given an mp3 filename, return the corresponding best alignment .mid name according to the .pdf present '''
        base, _ = os.path.splitext(filename)
        if os.path.exists(os.path.join(aligned_path, base + '.pdf')):
            return filename.replace('mp3', 'mid')
        n = 1
        while not os.path.exists(os.path.join(aligned_path, '{}.{}.pdf'.format(base, n))):
            n += 1
        return '{}.{}.mid'.format(base, n)
    def to_h5_path(filename):
        ''' Given an mp3 filename, returns the path to the corresponding -beats.npy file '''
        return os.path.join(base_data_path, get_data_folder(filename), 'msd', filename.replace('.mp3', '.h5'))
    def to_midi_path(filename):
        ''' Given an mp3 filename, returns the path to the corresponding midi file '''
        return os.path.join(base_data_path, get_data_folder(filename), midi_directory, to_numbered_mid(filename))

    # Load in list of files which were aligned correctly, and the start/end times of the good alignment
    files, start_times, end_times = load_results(tsv_path)

    for filename, start_time, end_time in zip(files, start_times, end_times):
        # Load in MSD hdf5 file
        h5 = hdf5_getters.open_h5_file_read(to_h5_path(filename))
        # Load in beat times from MSD
        beats = hdf5_getters.get_beats_start(h5)
        # Get indices which fall within the range of correct alignment
        time_mask = np.logical_and(beats > start_time, beats < end_time)
        beats = beats[time_mask]
        # and beat-synchronous feature matrices, within the time range of correct alignment
        chroma = beat_aligned_feats.get_btchromas(h5)[:, time_mask]
        timbre = beat_aligned_feats.get_bttimbre(h5)[:, time_mask]
        loudness = beat_aligned_feats.get_btloudnessmax(h5)[:, time_mask]
        h5.close()
        # Stack it
        msd_features = np.vstack([chroma, timbre, loudness])
        if np.isnan(msd_features).any():
            print filename
            continue
        # Load in pretty midi object
        pm = pretty_midi.PrettyMIDI(midi.read_midifile(to_midi_path(filename)))
        # Construct piano roll, aligned to the msd beat times
        piano_roll = pm.get_piano_roll(times=beats)
        # Ignore notes below 36 and above 84
        piano_roll = piano_roll[36:84, :]
        # Write out
        np.save(os.path.join(output_path, filename.replace('.mp3', '-msd.npy')), msd_features)
        np.save(os.path.join(output_path, filename.replace('.mp3', '-midi.npy')), piano_roll)

