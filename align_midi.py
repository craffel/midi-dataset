# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import librosa
import scipy.stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
import copy

# <codecell>

def dpmod(M, pen=None):
    '''
    Use dynamic programming to find a min-cost path through matrix M.
    
    Input:
        M - Matrix to find path through
        pen - additional cost for for (0,1) and (1,0) steps, default None which means np.median(M)
    Output:
        p, q - State sequence
        score - DP score
    '''
    
    # Set penality = median(M) if none was provided
    if pen is None:
        pen = np.median(M)
    
    # Matrix of local costs, initialized to input matrix
    D = np.copy(M)
    
    # Store the traceback
    phi = np.zeros(D.shape)
    # Handle first row/column, where we force back to the beginning
    D[0, :] = np.cumsum(D[0, :]) + np.arange(D.shape[1])*pen
    D[:, 0] = np.cumsum(D[:, 0]) + np.arange(D.shape[0])*pen
    phi[0, :] = 2
    phi[:, 0] = 1
    phi[0, 0] = 0   
    
    for i in xrange(D.shape[0] - 1): 
        for j in xrange(D.shape[1] - 1):
            # The possible locations we can move to, weighted by penalty score
            next_moves = [D[i, j], pen + D[i, j + 1], pen + D[i + 1, j]]
            # Choose the lowest cost
            tb = np.argmin(next_moves)
            dmin = next_moves[tb]
            # Add in the cost
            D[i + 1, j + 1] = D[i + 1, j + 1] + dmin
            # Store the traceback
            phi[i + 1, j + 1] = tb
    
    # Traceback from lowest-cost point on bottom or right edge
    i = np.argmin(D[:, -1])
    j = np.argmin(D[-1, :])
    if D[i, -1] < D[-1, j]:
        j = D.shape[1] - 1
    else:
        i = D.shape[0] - 1
    
    # Score is the final score of the best path
    score = D[i, j]
    
    # These vectors will give the lowest-cost path
    p = np.array([i])
    q = np.array([j])
    
    # Until we reach an edge
    while i > 0 or j > 0:
        # If the tracback matrix indicates a diagonal move...
        if phi[i, j] == 0:
            i = i - 1
            j = j - 1
        # Horizontal move...
        elif phi[i, j] == 1:
            i = i - 1
        # Vertical move...
        elif phi[i, j] == 2:
            j = j - 1
        # Add these indices into the path arrays
        p = np.append(i, p)
        q = np.append(j, q)
        
    # Normalize score
    score = score/q.shape[0]
    
    return p, q, score

# <codecell>

def maptimes(t, intime, outtime):
    '''
    map the times in t according to the mapping that each point in intime corresponds to that value in outtime
    2008-03-20 Dan Ellis dpwe@ee.columbia.edu
    
    Input:
        t - list of times to map
        intimes - original times
        outtime - mapped time
    Output:
        u - mapped version of t
    '''

    # Make sure both time ranges start at or before zero
    pregap = max(intime[0], outtime[0])
    intime = np.append(intime[0] - pregap, intime)
    outtime = np.append(outtime[0] - pregap, outtime)
    
    # Make sure there's a point beyond the end of both sequences
    din = np.diff(np.append(intime, intime[-1] + 1))
    dout = np.diff(np.append(outtime, outtime[-1] + 1))
    
    # Decidedly faster than outer-product-array way
    u = np.array(t)
    for i in xrange(t.shape[0]):
      ix = -1 + np.min(np.append(np.flatnonzero(intime > t[i]), outtime.shape[0]))
      # offset from that time
      dt = t[i] - intime[ix];
      # perform linear interpolation
      u[i] = outtime[ix] + (dt/din[ix])*dout[ix]
    return u

# <codecell>

def midi_to_cqt(midi, method=None, fs=22050, hop=512):
    '''
    Feature extraction routine for midi data, converts to a drum-free, percussion-suppressed CQT.
    
    Input:
        midi - pretty_midi.PrettyMIDI object
        method - synthesis method to pass to the midi object's synthesize method
        fs - sampling rate to synthesize audio at, default 22050
        hop - hop length for cqt, default 512
    Output:
        midi_gram - Simulated CQT of the midi data
    '''
    # Create a copy of the midi object
    midi_no_drums = copy.deepcopy(midi)
    # Remove the drums
    for n, instrument in enumerate(midi_no_drums.instruments):
        if instrument.is_drum:
            del midi_no_drums.instruments[n]
    # Synthesize the MIDI using the supplied method
    midi_audio = midi_no_drums.synthesize(fs=fs, method=method)
    # Use the harmonic part of the signal
    H, P = librosa.decompose.hpss(librosa.stft(midi_audio))
    midi_audio_harmonic = librosa.istft(H)
    # Compute log frequency spectrogram of audio synthesized from MIDI
    midi_gram = np.abs(librosa.cqt(y=midi_audio_harmonic,
                                   sr=fs,
                                   hop_length=hop,
                                   fmin=librosa.midi_to_hz(36),
                                   n_bins=60,
                                   tuning=0.0))**2
    return midi_gram

# <codecell>

def audio_to_cqt_and_onset_strength(audio, fs=22050, hop=512):
    '''
    Feature extraction for audio data.
    Gets a power CQT of harmonic component and onset strength signal of percussive.
    
    Input:
        midi - pretty_midi.PrettyMIDI object
        fs - sampling rate to synthesize audio at, default 22050
        hop - hop length for cqt, default 512, onset strength hop will be 1/4 of this
    Output:
        audio_gram - CQT of audio data
        audio_onset_strength - onset strength signal
    '''
    # Use harmonic part for gram, percussive part for onsets
    H, P = librosa.decompose.hpss(librosa.stft(audio))
    audio_harmonic = librosa.istft(H)
    audio_percussive = librosa.istft(P)
    # Compute log-frequency spectrogram of original audio
    audio_gram = np.abs(librosa.cqt(y=audio_harmonic,
                                    sr=fs,
                                    hop_length=hop,
                                    fmin=librosa.midi_to_hz(36),
                                    n_bins=60))**2
    # Beat track the audio file at 4x the hop rate
    audio_onset_strength = librosa.onset.onset_strength(audio_percussive, hop_length=hop/4, sr=fs)
    return audio_gram, audio_onset_strength

# <codecell>

def midi_beat_track(midi, fs=22050, hop=512.):
    '''
    Perform midi beat tracking and force the tempo to be high
    
    Input:
        midi - pretty_midi.PrettyMIDI object
        fs - sample rate to sample beats with
        hop - hop size to sample beats with
    Output:
        midi_beats - np.array of beat times, in frames, with sample rate fs and hop size 512
        midi_tempo - tempo, at least 240 bpm
    '''
    # Estimate MIDI beat times
    midi_beats = np.array(midi.get_beats()*fs/hop, dtype=np.int)
    # Estimate the MIDI tempo
    midi_tempo = 60.0/np.mean(np.diff(midi.get_beats()))
    # Make tempo faster for better temporal resolution
    scale = 1
    while midi_tempo < 240:
        midi_tempo *= 2
        scale *= 2
    # Interpolate the beats to match the higher tempo
    midi_beats = np.array(np.interp(np.linspace(0, scale*(midi_beats.shape[0] - 1), scale*(midi_beats.shape[0] - 1) + 1),
                                    np.linspace(0, scale*(midi_beats.shape[0] - 1), midi_beats.shape[0]),
                                    midi_beats), dtype=np.int)
    return midi_beats, midi_tempo

# <codecell>

def post_process_cqt(gram, beats):
    '''
    Given a power CQT, beat-synchronize it, take log, and normalize
    
    Input:
        gram - np.ndarray, power CQT
        beats - np.ndarray, beat locations in frame number
    Output:
        gram_normalized - CQT, normalized and beat-synchronized
    '''
    # Truncate to length of audio
    truncated_beats = beats[beats < gram.shape[1]]
    # Synchronize the log-fs gram with MIDI beats
    synchronized_gram = librosa.feature.sync(gram, truncated_beats)[:, 1:]
    # Compute log-amplitude spectrogram
    log_gram = librosa.logamplitude(synchronized_gram, ref_power=synchronized_gram.max())
    # Normalize columns and return
    return librosa.util.normalize(log_gram, axis=0)

# <codecell>

def adjust_midi(midi, original_times, new_times):
    '''
    Wrapper function to adjust all time locations in a midi object using maptimes
    
    Input:
        midi - pretty_midi.PrettyMIDI object
        original_times - np.ndarray of reference times
        new_times - np.ndarray of times to map to
    Output:
        aligned_midi - midi object with its times adjusted
    '''
    # Get array of note-on locations and correct them
    note_ons = np.array([note.start for instrument in midi.instruments for note in instrument.events])
    aligned_note_ons = maptimes(note_ons, original_times, new_times)
    # Same for note-offs
    note_offs = np.array([note.end for instrument in midi.instruments for note in instrument.events])
    aligned_note_offs = maptimes(note_offs, original_times, new_times)
    # Same for pitch bends
    pitch_bends = np.array([bend.time for instrument in midi.instruments for bend in instrument.pitch_bends])
    aligned_pitch_bends = maptimes(pitch_bends, original_times, new_times)
    # Create copy (not doing this in place)
    midi_aligned = copy.deepcopy(midi)
    # Correct notes
    for n, note in enumerate([note for instrument in midi_aligned.instruments for note in instrument.events]):
        note.start = (aligned_note_ons[n] > 0)*aligned_note_ons[n]
        note.end = (aligned_note_offs[n] > 0)*aligned_note_offs[n]
    # Correct pitch changes
    for n, bend in enumerate([bend for instrument in midi_aligned.instruments for bend in instrument.pitch_bends]):
        bend.time = (aligned_pitch_bends[n] > 0)*aligned_pitch_bends[n]
    return midi_aligned

# <codecell>

if __name__ == '__main__':
    import midi
    import pretty_midi
    import glob
    import subprocess
    import joblib
    import os
    SF2_PATH = '../Performer Synchronization Measure/SGM-V2.01.sf2'
    OUTPUT_PATH = 'midi-aligned-additive-dpmod'
    BASE_PATH = 'data/sanity'
    if not os.path.exists(os.path.join(BASE_PATH, OUTPUT_PATH)):
        os.makedirs(os.path.join(BASE_PATH, OUTPUT_PATH))
        
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
        
    def get_all_midi_candidates(filename):
        ''' Given some/path/audio/file.mp3, return list of [some/path/midi/file.mid, some/path/midi/file.1.mid, etc] '''
        midi_filename = mp3_to_mid(filename)
        midi_candidates = [midi_filename]
        i = 1
        while os.path.exists('{}.{}.mid'.format(os.path.splitext(midi_filename)[0], i)):
            midi_candidates.append('{}.{}.mid'.format(os.path.splitext(midi_filename)[0], i))
            i += 1
        return midi_candidates

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
            audio_gram, audio_onset_strength = audio_to_cqt_and_onset_strength(audio, fs=fs)
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
                midi_gram = midi_to_cqt(m, SF2_PATH)
                if fs is None:
                        audio, fs = librosa.load(mp3_filename)
                # Get beats
                midi_beats, bpm = midi_beat_track(m, fs=fs)
                # Beat synchronize and normalize
                midi_gram = post_process_cqt(midi_gram, midi_beats)
                # Write out
                np.save(to_cqt_npy(midi_filename), midi_gram)        

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
            midi_beats, bpm = midi_beat_track(m)
            audio_beats = librosa.beat.beat_track(onsets=audio_onset_strength, hop_length=512/4, bpm=bpm)[1]/4
            # Beat-align and log/normalize the audio CQT
            audio_gram = post_process_cqt(audio_gram, audio_beats)
            
            # Plot log-fs grams
            plt.figure(figsize=(24, 24))
            ax = plt.subplot2grid((4, 2), (0, 0), colspan=2)
            plt.title('MIDI Synthesized')
            librosa.display.specshow(midi_gram,
                                     x_axis='frames',
                                     y_axis='cqt_note',
                                     fmin=librosa.midi_to_hz(36),
                                     fmax=librosa.midi_to_hz(96))
            ax = plt.subplot2grid((4, 2), (1, 0), colspan=2)
            plt.title('Audio data')
            librosa.display.specshow(audio_gram,
                                     x_axis='frames',
                                     y_axis='cqt_note',
                                     fmin=librosa.midi_to_hz(36),
                                     fmax=librosa.midi_to_hz(96))
            
            # Get similarity matrix
            similarity_matrix = scipy.spatial.distance.cdist(midi_gram.T, audio_gram.T, metric='cosine')
            # Get best path through matrix
            p, q, score = dpmod(similarity_matrix)
            # Store the score
            candidate_costs[n] = score
    
            # Plot similarity matrix and best path through it
            ax = plt.subplot2grid((4, 2), (2, 0), rowspan=2)
            plt.imshow(similarity_matrix.T,
                       aspect='auto',
                       interpolation='nearest',
                       cmap=plt.cm.gray)
            tight = plt.axis()
            plt.plot(p, q, 'r.', ms=.2)
            plt.axis(tight)
            plt.title('Similarity matrix and lowest-cost path, cost={}'.format(score))
            
            # Adjust MIDI timing
            m_aligned = adjust_midi(m, librosa.frames_to_time(midi_beats)[p], librosa.frames_to_time(audio_beats)[q])
            # Store this MIDI object
            candidate_aligned_midi[midi_filename] = m_aligned
            
            # Plot alignment
            ax = plt.subplot2grid((4, 2), (2, 1), rowspan=2)
            note_ons = np.array([note.start for instrument in m.instruments for note in instrument.events])
            aligned_note_ons = np.array([note.start for instrument in m_aligned.instruments for note in instrument.events])
            plt.plot(note_ons, aligned_note_ons - note_ons, '.')
            plt.xlabel('Original note location (s)')
            plt.ylabel('Shift (s)')
            plt.title('Corrected offset')
            
            if output:
                # Save the figure for all midi files, even the one that's not the best
                plt.savefig(midi_filename.replace('midi', OUTPUT_PATH).replace('.mid', '.pdf'))
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
            
    # Parallelization!
    mp3_glob = glob.glob(os.path.join(BASE_PATH, 'audio', '*.mp3'))
    joblib.Parallel(n_jobs=6)(joblib.delayed(create_npys)(filename) for filename in mp3_glob)
    joblib.Parallel(n_jobs=6)(joblib.delayed(align_one_file)(filename) for filename in mp3_glob)

