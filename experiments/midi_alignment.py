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

# <codecell>

SF2_PATH = '../../Performer Synchronization Measure/SGM-V2.01.sf2'
OUTPUT_PATH = 'midi-aligned-additive-dpmod'
BASE_PATH = '../data/sanity'
if not os.path.exists(os.path.join(BASE_PATH, OUTPUT_PATH)):
    os.makedirs(os.path.join(BASE_PATH, OUTPUT_PATH))

# <codecell>

# Utility functions for converting between filenames
def mp3_to_mid(filename):
    ''' Given some/path/audio/file.mp3, return some/path/midi/file.mid '''
    return filename.replace('audio', 'midi').replace('.mp3', '.mid')
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

def get_all_midi_candidates(filename):
    ''' Given some/path/audio/file.mp3, return list of [some/path/midi/file.mid, some/path/midi/file.1.mid, etc] '''
    midi_filename = mp3_to_mid(filename)
    midi_candidates = [midi_filename]
    i = 1
    while os.path.exists('{}.{}.mid'.format(os.path.splitext(midi_filename)[0], i)):
        midi_candidates.append('{}.{}.mid'.format(os.path.splitext(midi_filename)[0], i))
        i += 1
    return midi_candidates

# <codecell>

def create_npys(mp3_filename):
    '''
    Helper function to create cqt and beat npy files
    Will do nothing if the corresponding .npy files already exist
    
    Input:
        mp3_filename - Full path to an .mp3 file.  All other paths will be derived from this.
    '''
    fs = None
    if not os.path.exists(to_onset_strength_npy(mp3_filename)) or not os.path.exists(to_cqt_npy(mp3_filename)):        
        print "Creating CQT and onset strength signal for {}".format(os.path.split(mp3_filename)[1])
        # Don't need to load in audio multiple times
        audio, fs = librosa.load(mp3_filename)
        # Create audio CQT, which is just frame-wise power, and onset strength
        audio_gram, audio_onset_strength = align_midi.audio_to_cqt_and_onset_strength(audio, fs=fs)
        # Write out
        np.save(to_onset_strength_npy(mp3_filename), audio_onset_strength)
        np.save(to_cqt_npy(mp3_filename), audio_gram)  
    
    # Get a list of all MIDI candidate files to try
    midi_candidates = get_all_midi_candidates(mp3_filename)
    for n, midi_filename in enumerate(midi_candidates):
        # If there's an error for this MIDI File, skip it
        try:
            m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
        except:
            continue
        if not os.path.exists(to_cqt_npy(midi_filename)):      
            print "  Creating CQT for {}".format(os.path.split(midi_filename)[1])
            # Generate synthetic MIDI CQT
            midi_gram = align_midi.midi_to_cqt(m, SF2_PATH)
            if fs is None:
                audio, fs = librosa.load(mp3_filename)
            # Get beats
            midi_beats, bpm = align_midi.midi_beat_track(m, fs=fs)
            # Beat synchronize and normalize
            midi_gram = align_midi.post_process_cqt(midi_gram, midi_beats)
            # Write out
            np.save(to_cqt_npy(midi_filename), midi_gram)

# <codecell>

def align_one_file(mp3_filename, output=True):
    '''
    Helper function for aligning a single audio file to all candidate midi files.
    Only writes out the MIDI file with the best alignment.
    
    Input:
        mp3_filename - Full path to an .mp3 file.  All other paths will be derived from this.
        output - Whether or not to write output files, default True.  If False just does plotting.
    '''
    # Get a list of all MIDI candidate files to try
    midi_candidates = get_all_midi_candidates(mp3_filename)
    # Store the DTW cost of each candidate
    candidate_costs = np.zeros(len(midi_candidates))
    # Save the aligned MIDI for each candidate, we'll choose the best
    candidate_aligned_midi = {}
    for n, midi_filename in enumerate(midi_candidates):
        # Load in the corresponding midi file in the midi directory, and return if there is a problem loading it
        try:
            m = pretty_midi.PrettyMIDI(midi.read_midifile(midi_filename))
        except:
            # Store infinite cost so this doesn't get chosen
            candidate_costs[n] = np.inf
            # Skip to next
            continue
        
        print "Aligning {}".format(os.path.split(midi_filename)[1])
        
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
        # Store the score
        candidate_costs[n] = score
        
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
        # Store this MIDI object
        candidate_aligned_midi[midi_filename] = m_aligned
        
        # Plot alignment
        ax = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
        note_ons = np.array([note.start for instrument in m.instruments for note in instrument.events])
        aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.events])
        plt.plot(note_ons, aligned_note_ons - note_ons, '.')
        plt.xlabel('Original note location (s)')
        plt.ylabel('Shift (s)')
        plt.title('Corrected offset')
        
        if output:
            # Save the figure for all midi files, even the one that's not the best
            plt.savefig(midi_filename.replace('midi', OUTPUT_PATH).replace('.mid', '.pdf'))
        else:
            plt.show()
        plt.close()
    
    # If all MIDI files failed, just return
    if (candidate_costs == 0).all():
        return
    # Find the lowest-cost path
    n = np.argmin(candidate_costs)
    # Get the corresponding file path
    midi_filename = midi_candidates[n]
    # Get the aligned midi object
    m_aligned = candidate_aligned_midi[midi_filename]
    
    if output:
        # Write out the aligned file
        m_aligned.write(midi_filename.replace('midi', OUTPUT_PATH))
        # Load in the audio data (needed for writing out)
        audio, fs = librosa.load(mp3_filename, sr=None)
        # Synthesize the aligned midi
        midi_audio_aligned = m_aligned.synthesize(fs=fs, method=SF2_PATH)
        # Trim to the same size as audio
        if midi_audio_aligned.shape[0] > audio.shape[0]:
            midi_audio_aligned = midi_audio_aligned[:audio.shape[0]]
        else:
            midi_audio_aligned = np.pad(midi_audio_aligned,
                                        (0, audio.shape[0] - midi_audio_aligned.shape[0]),
                                        'constant')
        # Write out to temporary .wav file
        librosa.output.write_wav(mp3_filename.replace('audio', OUTPUT_PATH).replace('.mp3', '.wav'),
                                 np.vstack([midi_audio_aligned, audio]).T, fs)
        # Convert to mp3
        subprocess.check_output(['ffmpeg',
                         '-i',
                         mp3_filename.replace('audio', OUTPUT_PATH).replace('.mp3', '.wav'),
                         '-ab',
                         '128k',
                         '-y',
                         mp3_filename.replace('audio', OUTPUT_PATH)])
        # Remove temporary .wav file
        os.remove(mp3_filename.replace('audio', OUTPUT_PATH).replace('.mp3', '.wav'))

# <codecell>

# Parallelization!
mp3_glob = glob.glob(os.path.join(BASE_PATH, 'audio', '*.mp3'))
joblib.Parallel(n_jobs=7)(joblib.delayed(create_npys)(filename) for filename in mp3_glob)
joblib.Parallel(n_jobs=7)(joblib.delayed(align_one_file)(filename) for filename in mp3_glob)

