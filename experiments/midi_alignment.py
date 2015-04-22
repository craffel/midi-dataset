import numpy as np
import librosa
import pretty_midi
import joblib
import os
import align_midi
import sys
sys.path.append('..')
import itertools
import json
import alignment_utils

BASE_DATA_PATH = '../data/'
MIDI_PATH = 'clean_midi'
DATASETS = ['cal10k', 'cal500', 'uspop2002', 'msd']
OUTPUT_FOLDER = 'clean_midi_aligned'
AUDIO_FS = 22050
AUDIO_HOP = 512
MIDI_FS = 11025
MIDI_HOP = 256
NOTE_START = 36
N_NOTES = 48


def path_to_file(dataset, basename, extension):
    '''
    Returns the path to an actual file given the dataset, base file name,
    and file extension.  Assumes the dataset format of
    BASE_DATA_PATH/dataset/extension/basename.extension

    :parameters:
        - dataset : str
            Dataset, should be one of 'uspop2002', 'cal10k', 'cal500', etc.
        - basename : str
            Base name of the file
        - extension : str
            Extension of the file, e.g. mp3, h5, mid, npz, etc.

    :returns:
        - full_file_path : str
            Full path to the file in question.
    '''
    return os.path.join(BASE_DATA_PATH, dataset, extension,
                        '{}.{}'.format(basename, extension))


def check_subdirectories(filename):
    '''
    Checks that the subdirectories up to filename exist; if they don't, create
    them.

    :parameters:
        - filename : str
            Full path to file
    '''
    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])


def align_one_file(audio_filename, midi_filename, audio_features_filename=None,
                   midi_features_filename=None, output_midi_filename=None,
                   output_diagnostics_filename=None):
    '''
    Helper function for aligning a MIDI file to an audio file.

    :parameters:
        - audio_filename : str
            Full path to an audio file.
        - midi_filename : str
            Full path to a midi file.
        - audio_features_filename : str or NoneType
            Full path to pre-computed features for the audio file.
            If the file doesn't exist, features will be computed and saved.
            If None, force re-computation of the features and don't save.
        - midi_features_filename : str
            Full path to pre-computed features for the midi file.
            If the file doesn't exist, features will be computed and saved.
            If None, force re-computation of the features and don't save.
        - output_midi_filename : str or NoneType
            Full path to where the aligned .mid file should be written.
            If None, don't output.
        - output_diagnostics_filename : str or NoneType
            Full path to a file to write out diagnostic information (similarity
            matrix, best path, etc) in a .npz file.  If None, don't output.
    '''
    try:
        m = pretty_midi.PrettyMIDI(midi_filename)
    except Exception as e:
        print 'Could not parse {}: {}'.format(
            os.path.split(midi_filename)[1], e)
        return

    cache_midi_cqt = False

    # Cache MIDI CQT
    if midi_features_filename is None or \
            not os.path.exists(midi_features_filename):
        cache_midi_cqt = True
    else:
        try:
            # If a feature file was provided and exists, read it in
            features = np.load(midi_features_filename)
            midi_sync_gram = features['sync_gram']
            midi_beats = features['beats']
            midi_tempo = features['bpm']
        # If there was a problem reading, force re-cration
        except:
            cache_midi_cqt = True

    if cache_midi_cqt:
        # Generate synthetic MIDI CQT
        try:
            midi_audio = alignment_utils.fast_fluidsynth(m, MIDI_FS)
            midi_gram = librosa.cqt(
                midi_audio, sr=MIDI_FS, hop_length=MIDI_HOP,
                fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
            midi_beats, midi_tempo = alignment_utils.midi_beat_track(m)
            midi_sync_gram = alignment_utils.post_process_cqt(
                midi_gram, librosa.time_to_frames(
                    midi_beats, sr=MIDI_FS, hop_length=MIDI_HOP))
        except Exception as e:
            print "Error creating CQT for {}: {}".format(
                os.path.split(midi_filename)[1], e)
            return
        if midi_features_filename is not None:
            try:
                # Write out
                check_subdirectories(midi_features_filename)
                np.savez_compressed(
                    midi_features_filename, sync_gram=midi_sync_gram,
                    beats=midi_beats, bpm=midi_tempo)
            except Exception as e:
                print "Error writing npz for {}: {}".format(
                    os.path.split(midi_filename)[1], e)
                return

    cache_audio_cqt = False
    features = None

    if audio_features_filename is None or \
            not os.path.exists(audio_features_filename):
        cache_audio_cqt = True
    else:
        # If a feature file was provided and exists, read it in
        try:
            features = np.load(audio_features_filename)
            audio_gram = features['gram']
        # If there was a problem reading, force re-cration
        except:
            cache_audio_cqt = True

    # Cache audio CQT
    if cache_audio_cqt:
        try:
            audio, fs = librosa.load(audio_filename, sr=AUDIO_FS)
            audio_gram = librosa.cqt(
                audio, sr=fs, hop_length=AUDIO_HOP,
                fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
        except Exception as e:
            print "Error creating CQT for {}: {}".format(
                os.path.split(audio_filename)[1], e)
            return
        if audio_features_filename is not None:
            try:
                # Write out
                check_subdirectories(audio_features_filename)
                if features is not None:
                    np.savez_compressed(audio_features_filename,
                                        gram=audio_gram, **features)
                else:
                    np.savez_compressed(audio_features_filename,
                                        gram=audio_gram)
            except Exception as e:
                print "Error writing npz for {}: {}".format(
                    os.path.split(audio_filename)[1], e)
                return

    try:
        # Compute onset envelope from CQT (for speed)
        onset_envelope = librosa.onset.onset_strength(
            S=audio_gram, aggregate=np.median)
        _, audio_beats = librosa.beat.beat_track(
            onset_envelope=onset_envelope, bpm=midi_tempo)
        audio_sync_gram = alignment_utils.post_process_cqt(
            audio_gram, audio_beats)
    except Exception as e:
        print "Error syncing CQT for {}: {}".format(
            os.path.split(audio_filename)[1], e)
        return

    try:
        # Get similarity matrix
        similarity_matrix = 1 - np.dot(midi_sync_gram, audio_sync_gram.T)
        # Get best path through matrix
        p, q, score = align_midi.dpmod(similarity_matrix)
        # Normalize score by score by mean sim matrix value within path chunk
        score /= similarity_matrix[p.min():p.max(), q.min():q.max()].mean()
    except Exception as e:
        print "Error performing DTW for {} and {}: {}".format(
            os.path.split(audio_filename)[1],
            os.path.split(midi_filename)[1], e)
        return

    # Write out the aligned file
    if output_midi_filename is not None:
        try:
            # Adjust MIDI timing
            m_aligned = align_midi.adjust_midi(
                m, midi_beats[p], librosa.frames_to_time(audio_beats)[q])
            check_subdirectories(output_midi_filename)
            m_aligned.write(output_midi_filename)
        except Exception as e:
            print "Error writing aligned .mid for {}: {}".format(
                os.path.split(midi_filename)[1], e)
            return

    if output_diagnostics_filename is not None:
        try:
            check_subdirectories(output_diagnostics_filename)
            np.savez_compressed(
                output_diagnostics_filename, p=p, q=q, score=score,
                audio_filename=os.path.abspath(audio_filename),
                midi_filename=os.path.abspath(midi_filename),
                audio_features_filename=os.path.abspath(
                    audio_features_filename),
                midi_features_filename=os.path.abspath(midi_features_filename),
                output_midi_filename=os.path.abspath(output_midi_filename),
                output_diagnostics_filename=os.path.abspath(
                    output_diagnostics_filename))
        except Exception as e:
            print "Error writing diagnostics for {} and {}: {}".format(
                os.path.split(audio_filename)[1],
                os.path.split(midi_filename)[1], e)
            return


# Create the output dir if it doesn't exist
output_path = os.path.join(BASE_DATA_PATH, OUTPUT_FOLDER)
if not os.path.exists(output_path):
    os.makedirs(output_path)

for dataset in DATASETS + [MIDI_PATH, OUTPUT_FOLDER]:
    if not os.path.exists(os.path.join(BASE_DATA_PATH, dataset, 'npz')):
        os.makedirs(os.path.join(BASE_DATA_PATH, dataset, 'npz'))

# Load in the js filelist for the MIDI dataset used
midi_js = os.path.join(BASE_DATA_PATH, MIDI_PATH, 'index.js')
with open(midi_js) as f:
    midi_list = json.load(f)

# Load in the js filelist for the MIDI dataset used
midi_js = os.path.join(BASE_DATA_PATH, MIDI_PATH, 'index.js')
with open(midi_js) as f:
    midi_list = json.load(f)
midi_list = dict((e['md5'], e) for e in midi_list)

file_lists = {}
for dataset in DATASETS:
    # Load in the json filelist for this audio dataset
    with open(os.path.join(BASE_DATA_PATH, dataset, 'index.js')) as f:
        file_list = json.load(f)
    # Allow looking up entries by string ID returned by whoosh
    if dataset == 'msd':
        file_lists[dataset] = dict((e['track_id'], e) for e in file_list)
    else:
        file_lists[dataset] = dict((str(n), e)
                                   for n, e in enumerate(file_list))

# Load in pairs file
with open('../file_lists/text_matches.js') as f:
    text_matches = json.load(f)
flattened_matches = sum([list(itertools.product(*match))
                         for match in text_matches], [])

pairs = [(midi_list[midi_md5], file_lists[dataset][id])
         for midi_md5, (dataset, id) in flattened_matches]

# Construct a list of MIDI-audio matches, which will be attempted alignments
pairs = []
for midi_md5, (dataset, id) in flattened_matches:
    # Populate aruments for each pair
    file_basename = file_lists[dataset][id]['path']
    output_basename = '{}_{}_{}'.format(dataset, id, midi_md5)
    audio_filename = path_to_file(
        dataset, file_basename, 'mp3')
    midi_filename = path_to_file(
        MIDI_PATH, midi_list[midi_md5]['path'], 'mid')
    audio_features_filename = path_to_file(
        dataset, file_basename, 'npz')
    midi_features_filename = path_to_file(
        MIDI_PATH, midi_list[midi_md5]['path'], 'npz')
    output_midi_filename = path_to_file(
        OUTPUT_FOLDER, output_basename, 'mid')
    output_diagnostics_filename = path_to_file(
        OUTPUT_FOLDER, output_basename, 'npz')
    pairs.append((audio_filename, midi_filename,
                  audio_features_filename, midi_features_filename,
                  output_midi_filename,
                  output_diagnostics_filename))

# Run alignment
joblib.Parallel(n_jobs=10, verbose=51)(joblib.delayed(align_one_file)(*args)
                                       for args in pairs)
