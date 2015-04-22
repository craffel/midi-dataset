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
import librosa
import alignment_utils
import joblib
import scipy.interpolate
AUDIO_FS = 22050
AUDIO_HOP = 512
MIDI_FS = 11025
MIDI_HOP = 256
NOTE_START = 36
N_NOTES = 48

# A DP score above this means the alignment is bad
SCORE_THRESHOLD = .78
BASE_DATA_PATH = '../data'
FILE_LIST_PATH = '../file_lists'

aligned_path = os.path.join(BASE_DATA_PATH, 'clean_midi_aligned')
output_path = os.path.join(BASE_DATA_PATH, 'hash_dataset')

if not os.path.exists(os.path.join(output_path, 'npz')):
    os.makedirs(os.path.join(output_path, 'npz'))


def process_one_file(diagnostics_file, dataset):
    # If the alignment failed and there was no diagnostics file, return
    if not os.path.exists(diagnostics_file):
        return
    diagnostics = np.load(diagnostics_file)
    score = diagnostics['score']
    # Skip bad alignments
    if score > SCORE_THRESHOLD:
        return
    try:
        # Load in MIDI data and extract beats
        pm = pretty_midi.PrettyMIDI(str(diagnostics['output_midi_filename']))
        audio_data, audio_fs = librosa.load(str(diagnostics['audio_filename']))
        audio_features = np.load(str(diagnostics['audio_features_filename']))
        if 'beats' not in audio_features:
            gram = audio_features['gram']
            # Compute onset envelope from CQT (for speed)
            onset_envelope = librosa.onset.onset_strength(
                S=gram, aggregate=np.median)
            bpm, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
            # Double the BPM and interpolate beat locations if BPM < 160
            while bpm < 240:
                beat_interp = scipy.interpolate.interp1d(
                    np.arange(0, 2*beats.shape[0], 2), beats)
                beats = beat_interp(
                    np.arange(2*beats.shape[0] - 1)).astype(int)
                bpm *= 2
            beats = librosa.frames_to_time(beats)
        else:
            beats = audio_features['beats']
        # Get indices which fall within the range of correct alignment
        start_time = min([n.start for i in pm.instruments for n in i.notes])
        end_time = min(pm.get_end_time(), beats.max(),
                    audio_data.shape[0]/float(audio_fs))
        time_mask = np.logical_and(beats >= start_time, beats <= end_time)
        beats = beats[time_mask]
        if beats.size == 0:
            return
        # Synthesize MIDI data and extract CQT
        midi_audio = alignment_utils.fast_fluidsynth(pm, MIDI_FS)
        midi_gram = librosa.cqt(
            midi_audio, sr=MIDI_FS, hop_length=MIDI_HOP,
            fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
        midi_sync_gram = alignment_utils.post_process_cqt(
            midi_gram, librosa.time_to_frames(beats, MIDI_FS, MIDI_HOP))
        # Extract audio CQT, synchronized to MIDI beats
        audio_gram = audio_features['gram']
        audio_sync_gram = alignment_utils.post_process_cqt(
            audio_gram, librosa.time_to_frames(beats, AUDIO_FS, AUDIO_HOP))
        # Write out matrices with a newaxis at front (for # of channels)
        output_filename = os.path.join(
            output_path, dataset, 'npz', os.path.basename(diagnostics_file))
        np.savez_compressed(
            output_filename, X=midi_sync_gram[np.newaxis],
            Y=audio_sync_gram[np.newaxis])
    except Exception as e:
        print "Error for {}: {}".format(diagnostics_file, e)
        return


def pair_to_path(pair):
    '''
    Convert a pair [midi_md5, dataset, id] to a diagnostics files path

    :parameters:
        - pair : list of str
            Three entry list of [midi_md5, dataset, id]

    :returns:
        - path : str
            Path to the diagnostics file
    '''
    midi_md5, dataset, id = pair
    return os.path.join(aligned_path, 'npz',
                        '{}_{}_{}.npz'.format(dataset, id, midi_md5))

for dataset in ['train', 'valid']:
    if not os.path.exists(output_path, dataset, 'npz'):
        os.makedirs(os.path.join(output_path, dataset, 'npz'))
    # Load in all train pairs
    with open(os.path.join(FILE_LIST_PATH,
                           '{}_pairs.csv'.format(dataset))) as f:
        train_pairs = [line.strip().split(',') for line in f]

    # Create hashing .npz file for each pair
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(process_one_file)(pair_to_path(pair), dataset)
        for pair in train_pairs)
