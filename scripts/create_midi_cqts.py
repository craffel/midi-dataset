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
        m = pretty_midi.PrettyMIDI(midi_filename)
        midi_audio = alignment_utils.fast_fluidsynth(m, MIDI_FS)
        midi_gram = librosa.cqt(
            midi_audio, sr=MIDI_FS, hop_length=MIDI_HOP,
            fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
        midi_beats, midi_tempo = alignment_utils.midi_beat_track(m)
        midi_sync_gram = alignment_utils.post_process_cqt(
            midi_gram, librosa.time_to_frames(
                midi_beats, sr=MIDI_FS, hop_length=MIDI_HOP))
        np.savez_compressed(
            output_filename, sync_gram=midi_sync_gram,
            beats=midi_beats, bpm=midi_tempo)
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
