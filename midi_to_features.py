# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import midi
import numpy as np
import os
import random
import string

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
    
    # get notes
    # Array for holding notes (1-127)
    noteMatrix = np.zeros((128,len(onset_strength)+2*fs))  
    print (len(onset_strength))
    
    for track in MIDIData:
        time = 0
        for event in track:
            # Increment time by the tick value of this event
            time += event.tick*tickScale
            
            # If it's a note on event, we'll update the indicator and record the note
            if event.name == 'Note On' and event.channel != 9 and event.velocity > 0:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = event.velocity
                #print '[' + str(event.pitch) + ']' + '[' + str(index) + ']' + ':' + str(event.velocity)
         
            if event.name == 'Note Off' and event.channel != 9:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = 0
                #print '[' + str(event.pitch) + ']' + '[' + str(index) + ']' + ':' + str(event.velocity)
            
            if event.name == 'Note On' and event.channel != 9 and event.velocity == 0:
                index = int(time * fs)
                noteMatrix[event.pitch][index:] = 0
            
    return noteMatrix, onset_strength, fs


# <codecell>

if __name__=='__main__':
    print 'midi.read_midifile(MIDIFile)'
    print 'get_onsets_and_notes(MIDIData)'

