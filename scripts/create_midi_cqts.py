'''
Create feature files for clean_midi dataset .mid files
'''
import numpy as np
import librosa
import pretty_midi
import sys
sys.path.append('../')
import alignment_utils
import os
import glob
import joblib

MIDI_FS = 11025
MIDI_HOP = 256
NOTE_START = 36
N_NOTES = 48

BASE_DATA_PATH = '../data'
midi_glob = os.path.join(BASE_DATA_PATH, 'clean_midi', 'mid', '*', '*.mid')


def mid_to_npz_path(midi_filename):
    path, filename = os.path.split(midi_filename)
    path, last_subdir = os.path.split(path)
    output_filename = os.path.join(
        BASE_DATA_PATH, 'clean_midi', 'npz', last_subdir,
        filename.replace('.mid', '.npz'))
    return output_filename


def process_one_file(midi_filename, skip=True):
    '''
    Load in midi data, compute features, and write out file

    :parameters:
        - midi_filename : str
            Full path to midi file
        - skip : bool
            Whether to skip creating the file when the npz already exists
    '''
    # npz files go in the 'npz' dir instead of 'mid'
    output_filename = mid_to_npz_path(midi_filename)
    # Skip files already created
    if skip and os.path.exists(output_filename):
        return
    try:
        pm = pretty_midi.PrettyMIDI(midi_filename)
        max_frame = int(pm.get_end_time()*MIDI_FS/MIDI_HOP)
        midi_gram = pm.get_piano_roll(times=librosa.frames_to_time(
            np.arange(max_frame), sr=MIDI_FS, hop_length=MIDI_HOP))
        midi_gram = midi_gram[NOTE_START:NOTE_START + N_NOTES]
        beats, tempo = alignment_utils.midi_beat_track(pm)
        midi_sync_gram = librosa.feature.sync(
            midi_gram,
            librosa.time_to_frames(beats, sr=MIDI_FS, hop_length=MIDI_HOP),
            pad=False)
        midi_sync_gram = midi_sync_gram.T
        midi_sync_gram = librosa.util.normalize(midi_sync_gram, norm=2, axis=1)
        np.savez_compressed(
            output_filename, sync_gram=midi_sync_gram,
            beats=beats, bpm=tempo)
    except Exception as e:
        print "Error processing {}: {}".format(midi_filename, e)


# Create all output paths first to avoid joblib issues
for midi_filename in glob.glob(midi_glob):
    output_directory = os.path.split(mid_to_npz_path(midi_filename))[0]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

joblib.Parallel(n_jobs=11, verbose=51)(
    joblib.delayed(process_one_file)(midi_filename)
    for midi_filename in glob.glob(midi_glob))
