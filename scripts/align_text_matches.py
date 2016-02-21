import numpy as np
import librosa
import pretty_midi
import joblib
import os
import sys
sys.path.append('..')
import itertools
import json
import feature_extraction
import djitw
import deepdish
import whoosh_search
import traceback

BASE_DATA_PATH = '../data/'
MIDI_FOLDER = 'clean_midi'
DATASETS = ['cal10k', 'cal500', 'uspop2002', 'msd']
RESULTS_PATH = '../results/'
OUTPUT_FOLDER = 'clean_midi_aligned'


def path_to_file(base_path, basename, extension):
    '''
    Returns the path to an actual file given the dataset, base file name,
    and file extension.  Assumes the dataset format of
    BASE_DATA_PATH/dataset/extension/basename.extension

    Parameters
    ----------
    base_path : str
        Base path, should be e.g. os.path.join(BASE_DATA_PATH, 'uspop2002')
    basename : str
        Base name of the file
    extension : str
        Extension of the file, e.g. mp3, h5, mid

    Returns
    -------
    full_file_path : str
        Full path to the file in question.
    '''
    return os.path.join(base_path, extension,
                        '{}.{}'.format(basename, extension))


def check_subdirectories(filename):
    '''
    Checks that the subdirectories up to filename exist; if they don't, create
    them.

    Parameters
    ----------
    filename : str
        Full path to file
    '''
    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])


def align_one_file(audio_filename, midi_filename, audio_features_filename=None,
                   midi_features_filename=None, output_midi_filename=None,
                   output_diagnostics_filename=None,
                   additional_diagnostics=None):
    '''
    Helper function for aligning a MIDI file to an audio file.

    Parameters
    ----------
    audio_filename : str
        Full path to an audio file.
    midi_filename : str
        Full path to a midi file.
    audio_features_filename : str or None
        Full path to pre-computed features for the audio file.
        If the file doesn't exist, features will be computed and saved.
        If None, force re-computation of the features and don't save.
    midi_features_filename : str or None
        Full path to pre-computed features for the midi file.
        If the file doesn't exist, features will be computed and saved.
        If None, force re-computation of the features and don't save.
    output_midi_filename : str or None
        Full path to where the aligned .mid file should be written.
        If None, don't output.
    output_diagnostics_filename : str or None
        Full path to a file to write out diagnostic information (alignment
        score, best path, paths to files, etc) in a .h5 file.  If None, don't
        output.
    additional_diagnostics : dict or None
        Optional dictionary of additional diagnostic information to include
        in the diagnostics file.  If None, don't include.

    Returns
    -------
    p, q : np.ndarray
        Indices of the lowest-cost alignment between the audio and MIDI
    score : float
        Normalized DTW path distance
    '''
    # Skip when already processed
    if (output_diagnostics_filename is not None
            and os.path.exists(output_diagnostics_filename)):
        return

    try:
        m = pretty_midi.PrettyMIDI(midi_filename)
    except Exception as e:
        print 'Could not parse {}: {}'.format(
            os.path.split(midi_filename)[1], traceback.format_exc(e))
        return

    midi_features = {}

    # If a feature file was provided and the file exists, try to read it in
    if (midi_features_filename is not None and
            os.path.exists(midi_features_filename)):
        try:
            # If a feature file was provided and exists, read it in
            midi_features = deepdish.io.load(midi_features_filename)
        # If there was a problem reading, force re-cration
        except Exception as e:
            print "Error reading {}: {}".format(
                midi_features_filename, traceback.format_exc(e))
            midi_features = {}

    if not midi_features:
        # Generate synthetic MIDI CQT
        try:
            midi_features['gram'] = feature_extraction.midi_cqt(m)
        except Exception as e:
            print "Error creating CQT for {}: {}".format(
                os.path.split(midi_filename)[1], traceback.format_exc(e))
            return
        if midi_features_filename is not None:
            try:
                # Write out
                check_subdirectories(midi_features_filename)
                deepdish.io.save(
                    midi_features_filename, midi_features)
            except Exception as e:
                print "Error writing {}: {}".format(
                    os.path.split(midi_filename)[1], traceback.format_exc(e))
                return

    audio_features = {}

    # If a feature file was provided and the file exists, try to read it in
    if (audio_features_filename is not None and
            os.path.exists(audio_features_filename)):
        # If a feature file was provided and exists, read it in
        try:
            audio_features = deepdish.io.load(audio_features_filename)
        # If there was a problem reading, force re-cration
        except Exception as e:
            print "Error reading {}: {}".format(
                audio_features_filename, traceback.format_exc(e))
            audio_features = {}

    # Cache audio CQT
    if not audio_features:
        try:
            # Read in audio data
            audio, fs = librosa.load(
                audio_filename, sr=feature_extraction.AUDIO_FS)
            # Compute audio cqt
            audio_features['gram'] = feature_extraction.audio_cqt(audio)
        except Exception as e:
            print "Error creating CQT for {}: {}".format(
                os.path.split(audio_filename)[1], traceback.format_exc(e))
            return
        if audio_features_filename is not None:
            try:
                # Write out
                check_subdirectories(audio_features_filename)
                deepdish.io.save(audio_features_filename, audio_features)
            except Exception as e:
                print "Error writing {}: {}".format(
                    os.path.split(audio_filename)[1], traceback.format_exc(e))
                return

    try:
        # Check that the distance matrix will not be too big before computing
        size = midi_features['gram'].shape[0]*audio_features['gram'].shape[0]
        # If > 1 GB, skip
        if (size*64/8e9 > 2):
            print (
                "Distance matrix for {} and {} would be {} GB because the "
                "CQTs have shape {} and {}".format(
                    os.path.split(audio_filename)[1],
                    os.path.split(midi_filename)[1],
                    size*64/8e9, audio_features['gram'].shape[0],
                    midi_features['gram'].shape[0]))
            return

        # Get distance matrix
        distance_matrix = 1 - np.dot(
            midi_features['gram'], audio_features['gram'].T)
        # Non-diagonal additive path penalty is the median of the sim mtx
        add_pen = np.median(distance_matrix)
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
        # The confidence score is a normalized DTW distance, which
        # approximately follows in the range [.5, 1.] with .5 meaning a very
        # good alignment.  This maps the scores from [0., 1.] where 1. means a
        # very good alignment.
        score = np.clip(2*(1 - score), 0, 1)
    except Exception as e:
        print "Error performing DTW for {} and {}: {}".format(
            os.path.split(audio_filename)[1],
            os.path.split(midi_filename)[1],
            traceback.format_exc(e))
        return

    # Write out the aligned file
    if output_midi_filename is not None:
        try:
            # Adjust MIDI timing
            midi_frame_times = feature_extraction.frame_times(
                midi_features['gram'])
            audio_frame_times = feature_extraction.frame_times(
                audio_features['gram'])
            m.adjust_times(midi_frame_times[aligned_midi_indices],
                           audio_frame_times[aligned_audio_indices])
            check_subdirectories(output_midi_filename)
            m.write(output_midi_filename)
        except Exception as e:
            print "Error writing aligned .mid for {}: {}".format(
                os.path.split(midi_filename)[1], traceback.format_exc(e))
            return

    if output_diagnostics_filename is not None:
        try:
            check_subdirectories(output_diagnostics_filename)
            # Construct empty additional diagnostics dict when None was given
            if additional_diagnostics is None:
                additional_diagnostics = {}
            diagnostics = dict(
                aligned_midi_indices=aligned_midi_indices,
                aligned_audio_indices=aligned_audio_indices, score=score,
                audio_filename=os.path.abspath(audio_filename),
                midi_filename=os.path.abspath(midi_filename),
                audio_features_filename=os.path.abspath(
                    audio_features_filename),
                midi_features_filename=os.path.abspath(midi_features_filename),
                output_midi_filename=os.path.abspath(output_midi_filename),
                output_diagnostics_filename=os.path.abspath(
                    output_diagnostics_filename),
                **additional_diagnostics)
            deepdish.io.save(output_diagnostics_filename, diagnostics)
        except Exception as e:
            print "Error writing diagnostics for {} and {}: {}".format(
                os.path.split(audio_filename)[1],
                os.path.split(midi_filename)[1], traceback.format_exc(e))
            return
    return aligned_midi_indices, aligned_audio_indices, score


if __name__ == '__main__':
    # Create the output dir if it doesn't exist
    output_path = os.path.join(RESULTS_PATH, OUTPUT_FOLDER)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, 'mid')):
        os.makedirs(os.path.join(output_path, 'mid'))
    if not os.path.exists(os.path.join(output_path, 'h5')):
        os.makedirs(os.path.join(output_path, 'h5'))

    # Create feature file subdirectories if they don't exist
    for dataset in DATASETS + [MIDI_FOLDER]:
        if not os.path.exists(os.path.join(BASE_DATA_PATH, dataset, 'h5')):
            os.makedirs(os.path.join(BASE_DATA_PATH, dataset, 'h5'))

    file_lists = {}
    for dataset in DATASETS + [MIDI_FOLDER]:
        # Load in whoosh index
        index = whoosh_search.get_whoosh_index(
            os.path.join(BASE_DATA_PATH, dataset, 'index'))
        # Create all documents
        with index.searcher() as searcher:
            file_lists[dataset] = list(searcher.documents())
        # Allow lookup by ID
        file_lists[dataset] = dict((e['id'], e) for e in file_lists[dataset])

    # Load in pairs file
    with open(os.path.join(RESULTS_PATH, 'text_matches.js')) as f:
        text_matches = json.load(f)
    flattened_matches = sum([list(itertools.product(*match))
                            for match in text_matches], [])

    # Construct a list of MIDI-audio matches, which will be attempted
    pairs = []
    for midi_md5, (dataset, id) in flattened_matches:
        # Populate arguments for each pair
        file_basename = file_lists[dataset][id]['path']
        midi_basename = file_lists[MIDI_FOLDER][midi_md5]['path']
        output_basename = '{}_{}_{}'.format(dataset, id, midi_md5)
        audio_filename = path_to_file(
            os.path.join(BASE_DATA_PATH, dataset), file_basename, 'mp3')
        midi_filename = path_to_file(
            os.path.join(BASE_DATA_PATH, MIDI_FOLDER), midi_basename, 'mid')
        audio_features_filename = path_to_file(
            os.path.join(BASE_DATA_PATH, dataset), file_basename, 'h5')
        midi_features_filename = path_to_file(
            os.path.join(BASE_DATA_PATH, MIDI_FOLDER), midi_basename, 'h5')
        output_midi_filename = path_to_file(
            output_path, output_basename, 'mid')
        output_diagnostics_filename = path_to_file(
            output_path, output_basename, 'h5')
        additional_diagnostics = {
            'audio_dataset': dataset, 'audio_id': id, 'midi_md5': midi_md5}
        pairs.append((audio_filename, midi_filename, audio_features_filename,
                      midi_features_filename, output_midi_filename,
                      output_diagnostics_filename, additional_diagnostics))

    # Run alignment
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(align_one_file)(*args) for args in pairs)
