'''
Given alignment results, create a dataset for hashing using those alignments
which were sucessful (based on simple thresholding of dynamic programming
score) consisting of aligned CQTs from audio and synthesized MIDI
'''

import sys
sys.path.append('..')
import numpy as np
import os
import pretty_midi
import joblib
import feature_extraction
import deepdish
import traceback
import librosa

# A DP score below this means the alignment is bad
SCORE_THRESHOLD = .5
RESULTS_PATH = '../results'


def process_one_file(diagnostics_file, output_filename,
                     output_filename_unaligned, output_filename_piano_roll):
    # If the alignment failed and there was no diagnostics file, return
    if not os.path.exists(diagnostics_file):
        return
    diagnostics = deepdish.io.load(diagnostics_file)
    score = diagnostics['score']
    # Skip bad alignments
    if score < SCORE_THRESHOLD:
        return
    try:
        # Load in MIDI data
        pm_unaligned = pretty_midi.PrettyMIDI(
            str(diagnostics['midi_filename']))
        # Synthesize MIDI data and extract CQT
        midi_gram_unaligned = feature_extraction.midi_cqt(pm_unaligned)
        # Get audio CQT
        audio_features = deepdish.io.load(
            str(diagnostics['audio_features_filename']))
        audio_gram = audio_features['gram']
        audio_frame_times = feature_extraction.frame_times(audio_gram)
        # Write out unaligned MIDI CQT
        deepdish.io.save(output_filename_unaligned,
                         {'X': midi_gram_unaligned[np.newaxis],
                          'Y': audio_gram[np.newaxis]})
        # Load in MIDI data
        pm_aligned = pretty_midi.PrettyMIDI(
            str(diagnostics['output_midi_filename']))
        # Synthesize MIDI data and extract CQT
        midi_gram_aligned = feature_extraction.midi_cqt(pm_aligned)
        midi_frame_times = feature_extraction.frame_times(midi_gram_aligned)
        # Get indices which fall within the range of correct alignment
        start_time = min(
            n.start for i in pm_aligned.instruments for n in i.notes)
        end_time = min(pm_aligned.get_end_time(), midi_frame_times.max(),
                       audio_frame_times.max())
        if end_time <= start_time:
            return
        # Mask out the times within the aligned region
        audio_gram = audio_gram[np.logical_and(audio_frame_times >= start_time,
                                               audio_frame_times <= end_time)]
        midi_gram = midi_gram_aligned[
            np.logical_and(midi_frame_times >= start_time,
                           midi_frame_times <= end_time)]
        # Write out matrices with a newaxis at front (for # of channels)
        deepdish.io.save(
            output_filename, {'X': midi_gram[np.newaxis],
                              'Y': audio_gram[np.newaxis]})

        piano_roll = pm_aligned.get_piano_roll(times=midi_frame_times)
        # Only utilize the same notes which are used in the CQT
        piano_roll = piano_roll[
            feature_extraction.NOTE_START:
            feature_extraction.NOTE_START + feature_extraction.N_NOTES]
        # Transpose so that the first dimension is time
        piano_roll = piano_roll.T
        # L2 normalize columns
        piano_roll = librosa.util.normalize(piano_roll, norm=2, axis=1)
        # Mask out times within the aligned region
        piano_roll = piano_roll[
            np.logical_and(midi_frame_times >= start_time,
                           midi_frame_times <= end_time)]
        # Use float32 for Theano
        piano_roll = piano_roll.astype(np.float32)
        deepdish.io.save(
            output_filename_piano_roll, {'X': piano_roll[np.newaxis],
                                         'Y': audio_gram[np.newaxis]})
    except Exception as e:
        print "Error for {}: {}".format(
            diagnostics_file, traceback.format_exc(e))
        return


def pair_to_path(pair):
    '''
    Convert a pair [midi_md5, dataset, id] to a diagnostics files path

    Parameters
    ----------
    pair : list of str
        Three entry list of [midi_md5, dataset, id]

    Returns
    -------
    path : str
        Path to the diagnostics file
    '''
    midi_md5, dataset, id = pair
    return '{}_{}_{}.h5'.format(dataset, id, midi_md5)

if __name__ == '__main__':

    aligned_path = os.path.join(RESULTS_PATH, 'clean_midi_aligned', 'h5')

    for dataset in ['train', 'validate']:
        # Create output paths for this dataset split
        output_path = os.path.join(
            RESULTS_PATH, 'training_dataset', dataset, 'h5')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path_unaligned = os.path.join(
            RESULTS_PATH, 'training_dataset_unaligned', dataset, 'h5')
        if not os.path.exists(output_path_unaligned):
            os.makedirs(output_path_unaligned)
        output_path_piano_roll = os.path.join(
            RESULTS_PATH, 'training_dataset_piano_roll', dataset, 'h5')
        if not os.path.exists(output_path_piano_roll):
            os.makedirs(output_path_piano_roll)
        # Load in all pairs for this split
        pair_file = os.path.join(
            RESULTS_PATH, '{}_pairs.csv'.format(dataset))
        with open(pair_file) as f:
            pairs = [[line.strip().split(',')[n] for n in [1, 2, 0]]
                     for line in f]
        # Create hashing .h5 file for each pair
        joblib.Parallel(n_jobs=10, verbose=51)(
            joblib.delayed(process_one_file)(
                os.path.join(aligned_path, '{}_{}_{}.h5'.format(*pair)),
                os.path.join(output_path, '{}_{}_{}.h5'.format(*pair)),
                os.path.join(output_path_unaligned,
                             '{}_{}_{}.h5'.format(*pair)),
                os.path.join(output_path_piano_roll,
                             '{}_{}_{}.h5'.format(*pair)))
            for pair in pairs)
