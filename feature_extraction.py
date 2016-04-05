'''
Utilities for feature extraction.
'''

import numpy as np
import tempfile
import subprocess
import os
import pretty_midi
import librosa

AUDIO_FS = 22050
AUDIO_HOP = 1024
MIDI_FS = 11025
MIDI_HOP = 512
NOTE_START = 36
N_NOTES = 48


def fast_fluidsynth(m, fs):
    '''
    Faster fluidsynth synthesis using the command-line program
    instead of pyfluidsynth.

    Parameters
    ----------
    - m : pretty_midi.PrettyMIDI
        Pretty MIDI object
    - fs : int
        Sampling rate

    Returns
    -------
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
            ['fluidsynth', '-F', temp_wav.name, '-r', str(fs), sf2_path,
             temp_mid.name], stderr=devnull)
    # Load from temp wav file
    audio, _ = librosa.load(temp_wav.name, sr=fs)
    # Occasionally, fluidsynth pads a lot of silence on the end, so here we
    # crop to the length of the midi object
    audio = audio[:int(m.get_end_time() * fs)]
    # Close/delete temp files
    temp_mid.close()
    temp_wav.close()
    return audio


def midi_cqt(midi_object):
    '''
    Synthesize MIDI data, compute its constant-Q spectrogram, normalize, and
    log-scale it

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI data to create constant-Q spectrogram of.

    Returns
    -------
    midi_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrugram of synthesized MIDI
        data.
    '''
    # Synthesize MIDI object as audio data
    midi_audio = fast_fluidsynth(midi_object, MIDI_FS)
    # Compute CQT of the synthesized audio data
    midi_gram = librosa.cqt(
        midi_audio, sr=MIDI_FS, hop_length=MIDI_HOP,
        fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
    # L2-normalize and log-magnitute it
    return post_process_cqt(midi_gram)


def audio_cqt(audio_data, fs=AUDIO_FS):
    '''
    Compute some audio data's constant-Q spectrogram, normalize, and log-scale
    it

    Parameters
    ----------
    audio_data : np.ndarray
        Some audio signal.
    fs : int
        Sampling rate the audio data is sampled at, should be ``AUDIO_FS``.

    Returns
    -------
    midi_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrugram of synthesized MIDI
        data.
    '''
    # Compute CQT of the synthesized audio data
    audio_gram = librosa.cqt(
        audio_data, sr=fs, hop_length=AUDIO_HOP,
        fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
    # L2-normalize and log-magnitute it
    return post_process_cqt(audio_gram)


def post_process_cqt(gram):
    '''
    Normalize and log-scale a Constant-Q spectrogram

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram, constructed from ``librosa.cqt``.

    Returns
    -------
    log_normalized_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram.
    '''
    # Compute log amplitude
    gram = librosa.logamplitude(gram, ref_power=gram.max())
    # Transpose so that rows are samples
    gram = gram.T
    # and L2 normalize
    gram = librosa.util.normalize(gram, norm=2., axis=1)
    # and convert to float32
    return gram.astype(np.float32)


def frame_times(gram):
    '''
    Get the times corresponding to the frames in a spectrogram, which was
    created with one of the functions here.

    Parameters
    ----------
    gram : np.ndarray
        Spectrogram matrix.

    Returns
    -------
    times : np.ndarray
        Time, in seconds, of each frame in gram.
    '''
    # Note that because MIDI_FS = AUDIO_FS/2 and MIDI_HOP = AUDIO_HOP/2, using
    # AUDIO_FS and AUDIO_HOP here works whether it's a "MIDI" or "audio" cqt.
    return librosa.frames_to_time(
        np.arange(gram.shape[0]), AUDIO_FS, AUDIO_HOP)
