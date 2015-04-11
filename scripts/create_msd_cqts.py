'''
Create feature files for MSD 7digital 30 second clips
'''
import numpy as np
import librosa
import scipy.interpolate
import glob
import joblib
import os

BASE_DATA_PATH = '../data'
mp3_glob = os.path.join(BASE_DATA_PATH, 'msd', 'mp3', '*', '*', '*', '*.mp3')


def extract_features(audio_data):
    '''
    Feature extraction routine - gets beat-synchronous CQT, beats, and bpm

    :parameters:
        - audio_data : np.ndarray
            Audio samples at 22 kHz

    :returns:
        - cqt : np.ndarray
            Beat-synchronous CQT, four octaves, starting from note 36
        - beats : np.ndarray
            Beat locations, in seconds.  Beat tracking is done using CQT
        - bpm : float
            BPM.  If the estimated BPM is less than 160, it is doubled.
    '''
    gram = np.abs(librosa.cqt(
        audio_data, fmin=librosa.midi_to_hz(36), n_bins=48))
    # Compute onset envelope from CQT (for speed)
    onset_envelope = librosa.onset.onset_strength(S=gram, aggregate=np.median)
    bpm, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
    # Double the BPM and interpolate beat locations if BPM < 160
    if bpm < 160:
        beat_interp = scipy.interpolate.interp1d(
            np.arange(0, 2*beats.shape[0], 2), beats)
        beats = beat_interp(np.arange(2*beats.shape[0] - 1)).astype(int)
        bpm *= 2
    sync_gram = librosa.feature.sync(gram, beats)
    return sync_gram, librosa.frames_to_time(beats), bpm


def process_one_file(mp3_filename, skip=True):
    '''
    Load in an mp3, get the features, and write the features out

    :parameters:
        - mp3_filename : str
            Path to an mp3 file
        - skip : bool
            Whether to skip files when the npz already exists
    '''
    # npz files go in the 'npz' dir instead of 'mp3'
    output_filename = mp3_filename.replace(
        'clip.mp3', 'npz').replace('mp3', 'npz')
    # Skip files already created
    if skip and os.path.exists(output_filename):
        return
    try:
        audio_data, _ = librosa.load(mp3_filename)
        cqt, beats, bpm = extract_features(audio_data)
    except Exception as e:
        print "Error processing {}: {}".format(mp3_filename, e)
        return
    # Save as float32 to save space
    np.savez(output_filename, cqt=cqt.astype(np.float32),
             beats=beats.astype(np.float32), bpm=bpm)

# Create all output paths first to avoid joblib issues
for mp3_filename in glob.glob(mp3_glob):
    output_directory = os.path.split(mp3_filename.replace('mp3', 'npz'))[0]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

joblib.Parallel(n_jobs=10, verbose=50)(
    joblib.delayed(process_one_file)(mp3_filename)
    for mp3_filename in glob.glob(mp3_glob))
