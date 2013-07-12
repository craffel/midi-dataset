# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import midi
import numpy as np
import os
import random
import string
import librosa

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
    # Array for holding onset locations (in seconds)
    onsets = np.array([])
    # Array for velocities too
    velocities = np.array([])
    tickScale = 1
    foundTempo = 0
    # Find the tempo setting message
    for track in MIDIData:
        for event in track:
            if event.name == 'Set Tempo':
                if foundTempo:
                    print "Warning: Multiple tempi found."
                    break
                else:
                    tickScale = 60.0/(event.get_bpm()*MIDIData.resolution)
                    foundTempo = 1
    # Warn if no tempo message was found, and set tick scale to 1 (which is kind of useless)
    if np.isnan( tickScale ):
        print "Warning: No tempo found."
        tickScale = 1
        
    # Iterate through tracks and events
    for track in MIDIData:
        time = 0
        for event in track:
            # Increment time by the tick value of this event
            time += event.tick*tickScale
            # If it's a note on event, we'll record the onset time
            if event.name == 'Note On':
                # If it's not a note-off event masquerading as a note-on
                if event.velocity > 0:
                    # Check for duplicates before adding in
                    if not (onsets == time).any():
                        onsets = np.append( onsets, time )
                        velocities = np.append( velocities, event.velocity )
    
    
    # Define a sampling rate for the signal
    fs = 1000
    # Create an empty signal
    onset_strength = np.zeros( int(fs*onsets.max()) +1 )
     
    # Convert to samples, fill in onset values
    onsets_in_sample = onsets * fs
    for i in range(len(onsets_in_sample)):
        samp_pos = int(onsets_in_sample[i])
        onset_strength[samp_pos] = velocities[i]
           
    # Get beats
    bpm,beats = librosa.beat.beat_track(onsets = onset_strength)
    
    # Get notes
    noteMatrix = np.zeros((128,len(onset_strength)+2*fs))  
    for track in MIDIData:
        time = 0
        for event in track:
            # Increment time by the tick value of this event
            time += event.tick*tickScale
            
            # If it's a note on event, we'll update the indicator and record the note
            if event.name == 'Note On' and event.channel != 9 and event.velocity > 0:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = event.velocity
         
            if event.name == 'Note Off' and event.channel != 9:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = 0
            
            if event.name == 'Note On' and event.channel != 9 and event.velocity == 0:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = 0
            
    return noteMatrix, bpm, beats, fs


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
        chroma_matrix[note,:] = np.sum([noteMatrix[12*0+note,:], noteMatrix[12*1+note,:], noteMatrix[12*2+note,:],
                                        noteMatrix[12*3+note,:], noteMatrix[12*4+note,:], noteMatrix[12*5+note,:],
                                        noteMatrix[12*6+note,:], noteMatrix[12*7+note,:], noteMatrix[12*8+note,:],
                                        noteMatrix[12*9+note,:] ], axis=0)
    
        
    # Get beat-synchronized chroma matrix by taking the mean across each beat
    beatChroma = np.zeros((12,len(beats)+1))
    beatChroma[:,0] = np.mean(chroma_matrix[:,0:beats[0]-1],1)
    for i in range(len(beats)-1):
        beatChroma[:,i] = np.mean(chroma_matrix[:,beats[i]:beats[i+1]-1],1)
        
    beatChroma[:,-1] = np.mean(chroma_matrix[:,beats[i+1]:-1],1)
    
    # Normalize?
    
    
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
    beatChroma_normalize = beatChroma
    for col in range(beatChroma.shape[1]):
        chromaSum = np.sum(beatChroma[:,col])
        if chromaSum != 0:
            beatChroma_normalize[:,col] = beatChroma[:,col] / np.sum(beatChroma[:,col])
            
    return beatChroma_normalize

# <codecell>

if __name__=='__main__':
    print 'midi.read_midifile(MIDIFile)'
    print 'get_onsets_and_notes(MIDIData)'
    print 'get_beat_chroma(noteMatrix,beats)'
    print 'get_normalize_beatChroma(beatChroma)'

