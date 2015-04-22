'''
Utilities for alignment/feature extraction.
'''

import numpy as np
import tempfile
import subprocess
import os
import pretty_midi
import librosa


def fast_fluidsynth(m, fs):
    '''
    Faster fluidsynth synthesis using the command-line program
    instead of pyfluidsynth.

    :parameters:
        - m : pretty_midi.PrettyMIDI
            Pretty MIDI object
        - fs : int
            Sampling rate

    :returns:
        - midi_audio : np.ndarray
            Synthesized audio, sampled at fs
    '''
    # Write out temp mid file
    temp_mid = tempfile.NamedTemporaryFile()
    m.write(temp_mid.name)
    # Get path to temporary .wav file
    temp_wav = tempfile.NamedTemporaryFile()
    # Get path to default pretty_midi SF2
    sf2_path = os.path.join(os.path.dirname(pretty_midi.__file__),
                            pretty_midi.DEFAULT_SF2)
    # Make system call to fluidsynth
    with open(os.devnull, 'w') as devnull:
        subprocess.check_output(
            ['fluidsynth', '--fast-render={}'.format(temp_wav.name),
             '-r', str(fs), sf2_path, temp_mid.name], stderr=devnull)
    # Load from temp wav file
    audio, _ = librosa.load(temp_wav.name, sr=fs)
    # Close/delete temp files
    temp_mid.close()
    temp_wav.close()
    return audio


def midi_beat_track(midi):
    '''
    Perform midi beat tracking and force the tempo to be high

    Input:
        midi - pretty_midi.PrettyMIDI object
    Output:
        midi_beats - np.array of beat times in seconds
        midi_tempo - tempo, at least 240 bpm
    '''
    # Estimate MIDI beat times
    midi_beats = midi.get_beats()
    # Estimate the MIDI tempo
    midi_tempo = 60.0/np.mean(np.diff(midi_beats))
    # Make tempo faster for better temporal resolution
    scale = 1
    while midi_tempo < 240:
        midi_tempo *= 2
        scale *= 2
    # Interpolate the beats to match the higher tempo
    midi_beats = np.array(np.interp(
        np.linspace(0, scale*(midi_beats.shape[0] - 1),
                    scale*(midi_beats.shape[0] - 1) + 1),
        np.linspace(0, scale*(midi_beats.shape[0] - 1), midi_beats.shape[0]),
        midi_beats))
    return midi_beats, midi_tempo


def post_process_cqt(gram, beats):
    '''
    Beat-synchronize, normalize, and log-scale a CQT

    :parameters:
        - gram : np.ndarray
            Numpy array of audio samples, sampled at fs
        - beats : np.ndarray
            Beat locations in _frames_

    :returns:
        - sync_gram : np.ndarray
            Beat-synchronous CQT
    '''
    # Synchronize the CQT to the beats
    sync_gram = librosa.feature.sync(gram, beats, pad=False)
    # Also compute log amplitude
    sync_gram = librosa.logamplitude(sync_gram, ref_power=sync_gram.max())
    # Transpose so that rows are samples
    sync_gram = sync_gram.T
    # and L2 normalize
    sync_gram = librosa.util.normalize(sync_gram, norm=2., axis=1)
    return sync_gram
