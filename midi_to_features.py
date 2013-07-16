# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import midi
import numpy as np
import os
import random
import string
import librosa
import collections

# <codecell>

def get_onsets_and_notes( MIDIData ):
    '''
    Given a midi.Pattern, extract onset locations and their velocities
    
    Input:
        MIDIData - midi.Pattern object, via midi.read_midifile( MIDIFile )
    Output:
        noteMatrix - a matrix containing the velocity of the notes as a piano roll
        onset_strength - onset strength function
        fs - sampling rate of the onset strength function
    '''
    # Normalize to 120 bpm
    tickScale = 60.0/(120*MIDIData.resolution)
    
    # Dict of note events - keys are (track, channel, note), values are a list of [velocity, note on time, note off time]
    noteDict = collections.defaultdict(list)
    lastNoteOff = 0
    for trackNumber, track in enumerate( MIDIData ):
        time = 0
        for event in track:
            # Increment time by the tick value of this event
            time += event.tick*tickScale
            
            # For a note on event, always create a new note list in the dict
            if event.name == 'Note On' and event.channel != 9 and event.velocity > 0:
                noteDict[(trackNumber, event.channel, event.pitch)] += [[event.velocity, time]]
            
            elif (event.name == 'Note Off' and event.channel != 9) or (event.name == 'Note On' and event.channel != 9 and event.velocity == 0):
                # Was a note-on stored? (ignore spurious note-offs)
                if len( noteDict[(trackNumber, event.channel, event.pitch)] ) > 0:
                    # Has a note-off been stored already?
                    if len( noteDict[(trackNumber, event.channel, event.pitch)][-1] ) == 2:
                        noteDict[(trackNumber, event.channel, event.pitch)][-1] += [time]
                        # Remember the final note-off
                        if time > lastNoteOff:
                            lastNoteOff = time

    # Create note matrix
    fs = 1000
    noteMatrix = np.zeros( (128, fs*(lastNoteOff + 1)), dtype=np.int16 )
    firstNoteOn = np.inf
    for (track, channel, note), events in noteDict.items():
        for event in events:
            # Make sure a note off was recorded
            if len( event ) > 2: 
                velocity, start, end = event
                noteMatrix[note, int(start*fs):int(end*fs)] += velocity
                if start < firstNoteOn:
                    firstNoteOn = start
    
    # Create artificial beat locations
    beats = np.arange( firstNoteOn*fs, noteMatrix.shape[1], .5*fs )
    
    return noteMatrix, beats, fs

# <codecell>

def get_beat_chroma(noteMatrix,beats) :
    '''
    Given a midi note spectrum and beats, extract beat-synchronized chromagram
    
    Input:
        noteMatrix - a 12 by n matrix representing the note distribution in the midi
        beats      - beats detected by librosa's beat tracker
    Output:
        beatChroma - a matrix representing beat-synchronized chromagram
    '''
    # Fold into one octave
    chroma_matrix = np.zeros((12,noteMatrix.shape[1]))
    for note in range(12):
        chroma_matrix[note, :] = np.sum( noteMatrix[note::12], axis=0 )
        
    # Get beat-synchronized chroma matrix by taking the mean across each beat
    beatChroma = np.zeros((12,len(beats)+1))
    beatChroma[:,0] = np.mean(chroma_matrix[:,0:beats[0]-1],1)
    for i in range(len(beats)-1):
        beatChroma[:,i] = np.mean(chroma_matrix[:,beats[i]:beats[i+1]-1],1)
        
    beatChroma[:,-1] = np.mean(chroma_matrix[:,beats[i+1]:-1],1)
    
    return beatChroma
    

# <codecell>

def get_normalize_beatChroma(beatChroma):
    '''
    Given beatChroma, normalize each time segment into sum 1
    
    Input:
        beatChroma - unnormalized beat-synchronized chromagram
    Output:
        beatChroma_normalize - normalized beat-synchronized chromagram
    '''
    colMax = beatChroma.max( axis = 0 )
    # Avoid divide by 0 by adding 1 when max is 0
    return beatChroma/(colMax + (colMax == 0))

# <codecell>

if __name__=='__main__':
    MIDIFile = 'Data/3565Hero.mid'
    MIDIData = midi.read_midifile(MIDIFile)
    noteMatrix, beats, fs = get_onsets_and_notes(MIDIData)
    beatChroma = get_beat_chroma(noteMatrix,beats)
    beatChroma = get_normalize_beatChroma(beatChroma)

    plt.figure( figsize=(24, 15) )
    plt.subplot( 311 )
    plt.axis( 'off' )
    plt.title( 'MIDI Transcription (with beats)' )
    plt.imshow( noteMatrix[36:96, 4000:20000], interpolation='nearest', aspect='auto', origin='lower', cmap=plt.cm.gray_r )
    plt.vlines( beats[beats < 20000] - 4000, -.5, 59.5 )
    plt.subplot( 312 )
    plt.axis( 'off' )
    plt.title( 'MIDI-synthesized Chromagram' )
    plt.imshow( beatChroma[:,:(beats[beats < 20000].shape[0])], interpolation='nearest', aspect='auto', origin='lower', cmap=plt.cm.gray_r )
    plt.subplot( 313 )
    plt.axis( 'off' )
    plt.title( 'MSD Chromagram' )
    import beat_aligned_feats
    msdChroma = beat_aligned_feats.get_btchromas( 'Data/TRJWCEA128F4273174(hero).h5' )
    plt.imshow( msdChroma[:,5:(beats[beats < 20000].shape[0]) + 5], interpolation='nearest', aspect='auto', origin='lower', cmap=plt.cm.gray_r )
    

