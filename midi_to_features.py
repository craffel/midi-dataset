# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import midi
import numpy as np
import os
import random
import string

# <codecell>

# Write out the unique time of note-ons
def get_onsets( MIDIFile ):
    MIDIData = midi.read_midifile( MIDIFile )
    # Array for holding onset locations (in seconds)
    onsets = np.array([])
    # Scale for MIDI tick numbers, to convert to seconds
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
    # Return onset array
    return onsets

