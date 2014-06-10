# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
import librosa
import midi
import pretty_midi
import glob
import subprocess
import joblib
import os
import sys
sys.path.append('../')
import align_midi
import scipy.io

# <codecell>

SF2_PATH = '../../Performer Synchronization Measure/SGM-V2.01.sf2'
OUTPUT_PATH = 'midi-aligned-additive-dpmod'
BASE_PATH = '../data/sanity'
if not os.path.exists(os.path.join(BASE_PATH, OUTPUT_PATH)):
    os.makedirs(os.path.join(BASE_PATH, OUTPUT_PATH))

# <codecell>

# Utility functions for converting between filenames
def to_cqt_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_cqt.npy '''
    return os.path.splitext(filename)[0] + '_cqt.npy'
def to_beats_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_beats.npy '''
    return os.path.splitext(filename)[0] + '_beats.npy' 
def to_onset_strength_npy(filename):
    ''' Given some/path/file.mid or .mp3, return some/path/file_onset_strength.npy '''
    return os.path.splitext(filename)[0] + '_onset_strength.npy' 

# <codecell>

def align_one_file(mp3_filename, midi_filename, output_midi_filename, output_diagnostics=True):
    '''
    Helper function for aligning a MIDI file to an audio file.
    
    :parameters:
        - mp3_filename : str
            Full path to a .mp3 file.
        - midi_filename : str
            Full path to a .mid file.
        - output_midi_filename : str
            Full path to where the aligned .mid file should be written.  If None, don't output.
        - output_diagnostics : bool
            If True, also output a .pdf of figures, a .mat of the alignment results,
            and a .mp3 of audio and synthesized aligned audio
    '''
    # Load in the corresponding midi file in the midi directory, and return if there is a problem loading it
    try:
        m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
    except:
        print "Error loading {}".format(midi_filename)
        return
        
    print "Aligning {}".format(os.path.split(midi_filename)[1])
    
    # Cache audio CQT and onset strength
    if not os.path.exists(to_onset_strength_npy(mp3_filename)) or not os.path.exists(to_cqt_npy(mp3_filename)):        
        print "Creating CQT and onset strength signal for {}".format(os.path.split(mp3_filename)[1])
        # Don't need to load in audio multiple times
        audio, fs = librosa.load(mp3_filename)
        # Create audio CQT, which is just frame-wise power, and onset strength
        audio_gram, audio_onset_strength = align_midi.audio_to_cqt_and_onset_strength(audio, fs=fs)
        # Write out
        np.save(to_onset_strength_npy(mp3_filename), audio_onset_strength)
        np.save(to_cqt_npy(mp3_filename), audio_gram)  

    # Cache MIDI CQT
    if not os.path.exists(to_cqt_npy(midi_filename)):      
        print "Creating CQT for {}".format(os.path.split(midi_filename)[1])
        # Generate synthetic MIDI CQT
        midi_gram = align_midi.midi_to_cqt(m, SF2_PATH)
        # Get beats
        midi_beats, bpm = align_midi.midi_beat_track(m)
        # Beat synchronize and normalize
        midi_gram = align_midi.post_process_cqt(midi_gram, midi_beats)
        # Write out
        np.save(to_cqt_npy(midi_filename), midi_gram)
            
    # Load in CQTs
    audio_gram = np.load(to_cqt_npy(mp3_filename))
    midi_gram = np.load(to_cqt_npy(midi_filename))
    # and audio onset strength signal
    audio_onset_strength = np.load(to_onset_strength_npy(mp3_filename))
    
    # Compute beats
    midi_beats, bpm = align_midi.midi_beat_track(m)
    audio_beats = librosa.beat.beat_track(onsets=audio_onset_strength, hop_length=512/4, bpm=bpm)[1]/4
    # Beat-align and log/normalize the audio CQT
    audio_gram = align_midi.post_process_cqt(audio_gram, audio_beats)
    
    # Plot log-fs grams
    plt.figure(figsize=(36, 24))
    ax = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    plt.title('MIDI Synthesized')
    librosa.display.specshow(midi_gram,
                             x_axis='frames',
                             y_axis='cqt_note',
                             fmin=librosa.midi_to_hz(36),
                             fmax=librosa.midi_to_hz(96))
    ax = plt.subplot2grid((4, 3), (1, 0), colspan=3)
    plt.title('Audio data')
    librosa.display.specshow(audio_gram,
                             x_axis='frames',
                             y_axis='cqt_note',
                             fmin=librosa.midi_to_hz(36),
                             fmax=librosa.midi_to_hz(96))
    
    # Get similarity matrix
    similarity_matrix = scipy.spatial.distance.cdist(midi_gram.T, audio_gram.T, metric='cosine')
    # Get best path through matrix
    p, q, score = align_midi.dpmod(similarity_matrix)
    
    # Plot distance at each point of the lowst-cost path
    ax = plt.subplot2grid((4, 3), (2, 0), rowspan=2)
    plt.plot([similarity_matrix[p_v, q_v] for p_v, q_v in zip(p, q)])
    plt.title('Distance at each point on lowest-cost path')

    # Plot similarity matrix and best path through it
    ax = plt.subplot2grid((4, 3), (2, 1), rowspan=2)
    plt.imshow(similarity_matrix.T,
               aspect='auto',
               interpolation='nearest',
               cmap=plt.cm.gray)
    tight = plt.axis()
    plt.plot(p, q, 'r.', ms=.2)
    plt.axis(tight)
    plt.title('Similarity matrix and lowest-cost path, cost={}'.format(score))
    
    # Adjust MIDI timing
    m_aligned = align_midi.adjust_midi(m, librosa.frames_to_time(midi_beats)[p], librosa.frames_to_time(audio_beats)[q])
    
    # Plot alignment
    ax = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
    note_ons = np.array([note.start for instrument in m.instruments for note in instrument.events])
    aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.events])
    plt.plot(note_ons, aligned_note_ons - note_ons, '.')
    plt.xlabel('Original note location (s)')
    plt.ylabel('Shift (s)')
    plt.title('Corrected offset')

    # Write out the aligned file
    if output_midi_filename is not None:
        m_aligned.write(output_midi_filename)
    
    if output_diagnostics:
        # Save the figures
        plt.savefig(output_midi_filename.replace('.mid', '.pdf'))
        # Load in the audio data (needed for writing out)
        audio, fs = librosa.load(mp3_filename, sr=None)
        # Synthesize the aligned midi
        midi_audio_aligned = m_aligned.fluidsynth(fs=fs, sf2_path=SF2_PATH)
        # Trim to the same size as audio
        if midi_audio_aligned.shape[0] > audio.shape[0]:
            midi_audio_aligned = midi_audio_aligned[:audio.shape[0]]
        else:
            midi_audio_aligned = np.append(midi_audio_aligned, np.zeros(audio.shape[0] - midi_audio_aligned.shape[0]))
        # Write out to temporary .wav file
        librosa.output.write_wav(output_midi_filename.replace('.mid', '.wav'),
                                 np.vstack([midi_audio_aligned, audio]).T, fs)
        # Convert to mp3
        subprocess.check_output(['ffmpeg',
                         '-i',
                         output_midi_filename.replace('.mid', '.wav'),
                         '-ab',
                         '128k',
                         '-y',
                         output_midi_filename.replace('.mid', '.mp3')])
        # Remove temporary .wav file
        os.remove(output_midi_filename.replace('.mid', '.wav'))
        # Save a .mat of the results
        scipy.io.savemat(output_midi_filename.replace('.mid', '.mat'),
                         {'similarity_matrix': similarity_matrix,
                          'p': p, 'q': q, 'score': score})
    # If we aren't outputting a .pdf, show the plot
    else:
        plt.show()
    plt.close()

# <codecell>

# Parallelization!
mp3_glob = sorted(glob.glob(os.path.join(BASE_PATH, 'audio', '*.mp3')))
midi_glob = sorted(glob.glob(os.path.join(BASE_PATH, 'midi', '*.mid')))
joblib.Parallel(n_jobs=7)(joblib.delayed(align_one_file)(mp3_filename,
                                                         midi_filename,
                                                         midi_filename.replace('midi', OUTPUT_PATH))
                                                         for mp3_filename, midi_filename in zip(mp3_glob, midi_glob))
