'''
Find valid matches based on the results of match.py and copy aligned and
non-aligned versions.
Call it like this:
    python assemble_aligned_matches.py ../results/unique_midi_matched/\*/\*.mpk
'''

import numpy as np
import pretty_midi
import sys
import traceback
import os
import msgpack
import djitw
import shutil
import glob
import deepdish
import joblib
import json
import collections
from align_text_matches import check_subdirectories
sys.path.append('..')
import feature_extraction

SCORE_THRESHOLD = 0.5
BASE_DATA_PATH = '../data'
MSD_H5_PATH = '/media/hdd1/data/MillionSong/data'
RESULTS_PATH = '../results'


def process_one_pair(midi_filename, mp3_filename, h5_filename,
                     unaligned_output_filename, aligned_output_filename,
                     mp3_output_filename, h5_output_filename):
    """
    Given a candidate MIDI-audio match, align the MIDI to the audio, then copy
    the unaligned and aligned MIDI if the score is high enough.

    Parameters
    ----------
    midi_filename : str
        Path to the MIDI file to align.
    mp3_filename : str
        Path to the audio file to align to.
    unaligned_output_filename : str
        Where to copy the unaligned MIDI file if the match was successful.
    aligned_output_filename : str
        Where to write the aligned MIDI file if the match was successful.
    mp3_output_filename : str
        Where to copy the mp3 file if the match was successful.
    h5_output_filename : str
        Where to copy the h5 file if the match was successful.
    """
    try:
        m = pretty_midi.PrettyMIDI(midi_filename)
    except Exception as e:
        print 'Could not parse {}: {}'.format(
            os.path.split(midi_filename)[1], traceback.format_exc(e))
        return
    try:
        midi_gram = feature_extraction.midi_cqt(m)
    except Exception as e:
        print "Error creating CQT for {}: {}".format(
            os.path.split(midi_filename)[1], traceback.format_exc(e))
        return
    # Construct path to pre-computed audio CQT path
    audio_features_filename = mp3_filename.replace('mp3', 'h5')
    try:
        audio_features = deepdish.io.load(audio_features_filename)
    except Exception as e:
        print "Error loading CQT for {}: {}".format(
            os.path.split(audio_features_filename)[1],
            traceback.format_exc(e))
        return
    # Check that the distance matrix will not be too big before computing
    size = midi_gram.shape[0] * audio_features['gram'].shape[0]
    # If > 1 GB, skip
    if (size * 64 / 8e9 > 2):
        print (
            "Distance matrix would be {} GB because the "
            "CQTs have shape {} and {}".format(
                size * 64 / 8e9, audio_features['gram'].shape[0],
                midi_gram.shape[0]))
        return
    # Get distance matrix
    distance_matrix = 1 - np.dot(midi_gram, audio_features['gram'].T)
    # Non-diagonal additive path penalty is the mean of the sim mtx
    # Note that we typically use a median here, but a mean is faster and
    # produces close enough results
    add_pen = np.mean(distance_matrix)
    # Get best path through matrix
    aligned_midi_indices, aligned_audio_indices, score = djitw.dtw(
        distance_matrix, gully=.96, additive_penalty=add_pen,
        inplace=False)
    # Normalize score by path length
    score /= float(len(aligned_midi_indices))
    # Normalize score by score by mean sim matrix value within path chunk
    score /= distance_matrix[
        aligned_midi_indices.min():aligned_midi_indices.max(),
        aligned_audio_indices.min():aligned_audio_indices.max()].mean()
    # If the match was successful
    if score > SCORE_THRESHOLD:
        # Try adjusting MIDI timing and writing out
        try:
            # Retrieve timing of frames in CQTs
            midi_frame_times = feature_extraction.frame_times(midi_gram)
            audio_frame_times = feature_extraction.frame_times(
                audio_features['gram'])
            # Adjust MIDI file timing
            m.adjust_times(midi_frame_times[aligned_midi_indices],
                           audio_frame_times[aligned_audio_indices])
            # Make sure all output paths exist and write out
            check_subdirectories(aligned_output_filename)
            m.write(aligned_output_filename)
        except Exception as e:
            print "Error adjusting and writing {}: {}".format(
                os.path.split(midi_filename)[1],
                traceback.format_exc(e))
            return
        # Assuming the above worked, all we have to do now is copy
        # Check/create all necessary subdirectores
        check_subdirectories(unaligned_output_filename)
        check_subdirectories(mp3_output_filename)
        check_subdirectories(h5_output_filename)
        # Copy all files
        shutil.copy(midi_filename, unaligned_output_filename)
        shutil.copy(mp3_filename, mp3_output_filename)
        try:
            shutil.copy(h5_filename, h5_output_filename)
        except Exception as e:
            print "Could not copy {}: {}".format(
                os.path.split(h5_filename)[1],
                traceback.format_exc(e))
            return
        # Return list of msd_id, midi_md5, score]
        prefix, midi_filename = os.path.split(aligned_output_filename)
        msd_id = os.path.split(prefix)[1]
        midi_md5 = os.path.splitext(midi_filename)[0]
        return [msd_id, midi_md5, score]


def msd_path(msd_id):
    """Converts e.g. TRABCD123456789 to A/B/C/TRABCD123456789"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

if __name__ == '__main__':
    # Use the first argument as the glob of things to look for (including a
    # single file)
    mpk_filenames = glob.glob(sys.argv[1])
    # Construct output paths
    unaligned_output_path = os.path.join(
        RESULTS_PATH, 'unique_midi_matched_unaligned')
    aligned_output_path = os.path.join(
        RESULTS_PATH, 'unique_midi_matched_aligned')
    mp3_output_path = os.path.join(RESULTS_PATH, 'unique_midi_matched_mp3')
    h5_output_path = os.path.join(RESULTS_PATH, 'unique_midi_matched_h5')
    # Create list of dicts of filename args to pass to process_one_pair
    match_filenames = []
    for mpk_filename in mpk_filenames:
        midi_md5 = os.path.splitext(os.path.split(mpk_filename)[1])[0]
        # Load in list of potential matches for this file
        with open(mpk_filename) as f:
            matches = msgpack.unpack(f)
        # Find MSD IDs which had a high-score match to this MIDI
        for msd_id, score in matches:
            if score >= SCORE_THRESHOLD:
                # Construct all necessary paths
                match_filenames.append(
                    {'midi_filename': os.path.join(
                        BASE_DATA_PATH, 'unique_midi', 'mid', midi_md5[0],
                        midi_md5 + '.mid'),
                     'mp3_filename': os.path.join(
                         BASE_DATA_PATH, 'msd', 'mp3',
                         msd_path(msd_id) + '.mp3'),
                     'h5_filename': os.path.join(
                         MSD_H5_PATH, msd_path(msd_id) + '.h5'),
                     'unaligned_output_filename': os.path.join(
                         unaligned_output_path, msd_path(msd_id),
                         midi_md5 + '.mid'),
                     'aligned_output_filename': os.path.join(
                         aligned_output_path, msd_path(msd_id),
                         midi_md5 + '.mid'),
                     'mp3_output_filename': os.path.join(
                         mp3_output_path, msd_path(msd_id) + '.mp3'),
                     'h5_output_filename': os.path.join(
                         h5_output_path, msd_path(msd_id) + '.h5')})
    # Run alignment
    results = joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(process_one_pair)(**kw) for kw in match_filenames)

    # Filter out failed results
    results = [r for r in results if r is not None]
    # Convert to dict-of-dicts, 1st key MSD ID, 2nd key MIDI MD5, value = score
    mapping = collections.defaultdict(dict)
    for r in results:
        mapping[r[0]][r[1]] = r[2]
    # Write out
    score_file = os.path.join(RESULTS_PATH, 'unique_midi_match_scores.json')
    with open(score_file, 'wb') as f:
        json.dump(mapping, f)
