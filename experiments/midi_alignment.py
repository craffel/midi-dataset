import numpy as np
import scipy.spatial.distance
import librosa
import pretty_midi
import joblib
import os
import align_midi
import scipy.io
import sys
sys.path.append('..')
import whoosh_search
import json

BASE_DATA_PATH = '../data/'
MIDI_PATH = 'clean_midi'
DATASETS = ['uspop2002', 'cal10k', 'cal500']
OUTPUT_FOLDER = 'clean_midi_aligned'
FS = 22050
CQT_HOP = 512
ONSET_HOP_DIVISOR = 4


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


def harmonic_log_cqt(audio, fs=FS, hop=CQT_HOP):
    '''
    Compute a log-scaled, haromonic component, column-normalized CQT.

    :parameters:
        - audio : np.ndarray
            Audio signal
        - fs : int
            Sampling rate
        - hop : int
            Hop size
    :returns:
        - cqt : np.ndarray
            Harmonic log-scaled normalized CQT
    '''
    H, P = librosa.decompose.hpss(librosa.stft(audio))
    audio_harmonic = librosa.istft(H)
    # Compute log-frequency spectrogram of original audio
    audio_gram = np.abs(librosa.cqt(y=audio_harmonic, sr=fs, hop_length=hop,
                                    fmin=librosa.midi_to_hz(36), n_bins=60))**2
    log_gram = librosa.logamplitude(audio_gram,
                                    ref_power=audio_gram.max())
    return librosa.util.normalize(log_gram, axis=0)


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
    except:
        print 'Could not parse {}'.format(os.path.split(midi_filename)[1])
        return

    print "Aligning {}".format(os.path.split(midi_filename)[1])

    # Cache audio CQT and onset strength
    if audio_features_filename is None or \
            not os.path.exists(audio_features_filename):
        print "Creating CQT for {}".format(
            os.path.split(audio_filename)[1])
        audio, fs = librosa.load(audio_filename, sr=FS)
        audio_gram = harmonic_log_cqt(audio, fs)
        if audio_features_filename is not None:
            # Write out
            check_subdirectories(audio_features_filename)
            np.savez_compressed(audio_features_filename,
                                gram=audio_gram)
    else:
        # If a feature file was provided and exists, read it in
        features = np.load(audio_features_filename)
        audio_gram = features['gram']

    # Cache MIDI CQT
    if midi_features_filename is None or \
            not os.path.exists(midi_features_filename):
        print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
        # Generate synthetic MIDI CQT
        audio = m.fluidsynth(fs=FS)
        midi_gram = harmonic_log_cqt(audio, FS)
        if midi_features_filename is not None:
            # Write out
            check_subdirectories(midi_features_filename)
            np.savez_compressed(midi_features_filename,
                                gram=midi_gram)
    else:
        # If a feature file was provided and exists, read it in
        features = np.load(midi_features_filename)
        midi_gram = features['gram']

    # Get similarity matrix
    similarity_matrix = scipy.spatial.distance.cdist(midi_gram.T, audio_gram.T,
                                                     metric='cosine')
    # Get best path through matrix
    p, q, score = align_midi.dpmod(similarity_matrix)

    # Write out the aligned file
    if output_midi_filename is not None:
        # Adjust MIDI timing
        m_aligned = align_midi.adjust_midi(
            m, librosa.frames_to_time(np.arange(midi_gram.shape[1]))[p],
            librosa.frames_to_time(np.arange(audio_gram.shape[1]))[q])
        check_subdirectories(output_midi_filename)
        m_aligned.write(output_midi_filename)

    if output_diagnostics_filename is not None:
        check_subdirectories(output_diagnostics_filename)
        np.savez_compressed(
            output_diagnostics_filename,
            p=p, q=q, score=score, audio_filename=audio_filename,
            midi_filename=midi_filename,
            audio_features_filename=audio_features_filename,
            midi_features_filename=midi_features_filename,
            output_midi_filename=output_midi_filename,
            output_diagnostics_filename=output_diagnostics_filename)


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

# Construct a list of MIDI-audio matches, which will be attempted alignments
alignment_matches = []
pairs = []
for dataset in DATASETS:
    # Load in the json filelist for this audio dataset
    with open(os.path.join(BASE_DATA_PATH, dataset, 'index.js')) as f:
        file_list = json.load(f)
    # Load the whoosh index for this dataset
    index_path = os.path.join(BASE_DATA_PATH, dataset, 'index')
    index = whoosh_search.get_whoosh_index(index_path)
    with index.searcher() as searcher:
        for midi_entry in midi_list:
            # Match each MIDI file entry against this dataset
            results = whoosh_search.search(searcher, index.schema,
                                           midi_entry['artist'],
                                           midi_entry['title'])
            if results is not None:
                for result in results:
                    if int(result[0]) > len(file_list):
                        print dataset, results
                    file_basename = file_list[int(result[0])]['path']
                    output_basename = '{}_{}_{}'.format(dataset, result[0],
                                                        midi_entry['md5'])
                    alignment_matches.append({
                        'dataset': dataset,
                        'dataset_id': result[0],
                        'midi_md5': midi_entry['md5'],
                        'audio_path': file_basename,
                        'midi_path': midi_entry['path'],
                        'output_path': output_basename})
                    audio_filename = path_to_file(
                        dataset, file_basename, 'mp3')
                    midi_filename = path_to_file(
                        MIDI_PATH, midi_entry['path'], 'mid')
                    audio_features_filename = path_to_file(
                        dataset, file_basename, 'npz')
                    midi_features_filename = path_to_file(
                        MIDI_PATH, midi_entry['path'], 'npz')
                    output_midi_filename = path_to_file(
                        OUTPUT_FOLDER, output_basename, 'mid')
                    output_diagnostics_filename = path_to_file(
                        OUTPUT_FOLDER, output_basename, 'npz')
                    pairs.append((audio_filename,
                                  midi_filename,
                                  audio_features_filename,
                                  midi_features_filename,
                                  output_midi_filename,
                                  output_diagnostics_filename))

json_out = os.path.join(BASE_DATA_PATH, OUTPUT_FOLDER, 'index.js')
with open(json_out, 'wb') as f:
    json.dump(alignment_matches, f, indent=4)

# Run alignment
joblib.Parallel(n_jobs=10)(joblib.delayed(align_one_file)(*args)
                          for args in pairs)
