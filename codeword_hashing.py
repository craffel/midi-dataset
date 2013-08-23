# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# codeword_hashing
# given a midi file, detect a list of songs that match the codeword of the midi file

# <codecell>

import numpy as np
import beat_aligned_feats
import os
import collections
import midi
import midi_to_features

# <codecell>

BITS_12 = 2**np.arange(0, 12)

# <codecell>

# Packing four 12-bit numbers into one 48-bit number
def pack4(x):
    
    def mask(z):
        return int(z) & 0x00000FFF
    
    return (mask(x[0]) << 36) | (mask(x[1]) << 24) | (mask(x[2]) << 12) | mask(x[3])

# <codecell>

# Getting the codeword using thresholding
def getCodeword(beatChroma,thres):
    beatChroma[beatChroma < thres] = 0
    beatChroma[beatChroma >= thres] = 1
    return BITS_12.dot(beatChroma)

# <codecell>

# Hashing the milliong song dataset into codewords, storing in defaultdict(set)

def load_MSD(MSD_DIR,THRES_MSD):
    my_hashing = collections.defaultdict(set)
    ct = 0
    for root, directory, f in os.walk(MSD_DIR):
        for song_id in f:
            src = os.path.join( root, song_id )
            if os.path.splitext(song_id)[1].lower() == '.h5':
                fullName = os.path.join(root,song_id)
                print fullName
                msd_beatChroma = beat_aligned_feats.get_btchromas(fullName)
                if msd_beatChroma is not None and not np.isnan(msd_beatChroma).any():
                    msd_codeword = getCodeword(msd_beatChroma,THRES_MSD)
                    for i in range(len(msd_codeword)-3):
                        subcode = msd_codeword[i:i+4]
                        my_hashing[pack4(subcode)].add(song_id)
                    ct += 1
                
                #if ct % 1000 == 0:
                #    print ct
    return my_hashing

# <codecell>

# Load the midi file and get the codeword array

def load_MIDI(midi_Name,THRES_MIDI):
    midiData = midi.read_midifile(midi_Name)
    noteMatrix, beats, fs = midi_to_features.get_onsets_and_notes(midiData)
    midi_beatChroma = midi_to_features.get_beat_chroma(noteMatrix,beats)
    midi_beatChroma = midi_to_features.get_normalize_beatChroma(midi_beatChroma)
    midi_codeword = getCodeword(midi_beatChroma,THRES_MIDI)
    return midi_codeword

# <codecell>

# Find a list of songs that match the 4-beat-based codeword

def find_song_id(midi_codeword, MSD_Hashing):
    song_count = collections.defaultdict(int)
    for i in range(len(midi_codeword)-3):
        subcode = midi_codeword[i:i+4]
        song_list = MSD_Hashing[pack4(subcode)]
        for elem in song_list:
            song_count[elem] += 1
    song_count_sort = sorted(song_count.items(), key=lambda t: t[1], reverse = True)
    return song_count_sort

