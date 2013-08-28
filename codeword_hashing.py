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

# Packing four 12-bit numbers into one 48-bit number
def pack4(x):
    
    def mask(z):
        return int(z) & 0x00000FFF
    
    return (mask(x[0]) << 36) | (mask(x[1]) << 24) | (mask(x[2]) << 12) | mask(x[3])

# <codecell>

# Getting the codeword using thresholding
BITS_12 = 2**np.arange(0, 12)
def getCodeword(beatChroma,thres):
    beatChroma[beatChroma < thres] = 0
    beatChroma[beatChroma >= thres] = 1
    # Dot product of the beatChroma and the BITS array to get a binary codeword for each beat
    return BITS_12.dot(beatChroma)

# <codecell>

# Hashing the milliong song dataset into codewords, storing in defaultdict(set)

def load_MSD(MSD_DIR,THRES_MSD):
    codeIndex = collections.defaultdict(set)
    #ct = 0
    # Loop through the MSD directory
    for root, directory, f in os.walk(MSD_DIR):
        for song_id in f:
            src = os.path.join( root, song_id )
            if os.path.splitext(song_id)[1].lower() == '.h5':
                fullName = os.path.join(root,song_id)
                # Get the beat Chroma of the song
                msd_beatChroma = beat_aligned_feats.get_btchromas(fullName)
                if msd_beatChroma is not None and not np.isnan(msd_beatChroma).any():
                    # Get the codewords of the song
                    msd_codeword = getCodeword(msd_beatChroma,THRES_MSD)
                    # For 4-codeword in codewords of the song
                    for i in range(len(msd_codeword)-3):
                        subcode = msd_codeword[i:i+4]
                        # Pack four 12-bit-codewords into a 48 bit binary number, and hash each codeword to the codeIndex
                        codeIndex[pack4(subcode)].add(song_id)
                    #ct += 1
                
                #if ct % 1000 == 0:
                    #print ct
    return codeIndex

# <codecell>

# Load the midi file and get the codeword array

def load_MIDI(midi_Name,THRES_MIDI):
    midiData = midi.read_midifile(midi_Name)
    noteMatrix, beats, fs = midi_to_features.get_onsets_and_notes(midiData)
    midi_beatChroma = midi_to_features.get_beat_chroma(noteMatrix,beats)
    midi_beatChroma = midi_to_features.get_normalize_beatChroma(midi_beatChroma)
    MIDI_codewords = getCodeword(midi_beatChroma,THRES_MIDI)
    return MIDI_codewords

# <codecell>

# A helper function 
# Given an array of midi codewords, for each 4-codeword-combination, find a list of matching-songs, and return a sorted song_id_count
def get_candidates(MIDI_codewords, codeIndex):
    song_id_count = collections.defaultdict(int)
    for i in range(len(midi_codeword)-3):
        subcode = midi_codeword[i:i+4]
        song_list = codeIndex[pack4(subcode)]
        for elem in song_list:
            song_id_count[elem] += 1
    song_id_count_sort = sorted(song_id_count.items(), key=lambda t: t[1], reverse = True)
    return song_id_count_sort

# <codecell>

def find_MIDI (codeIndex, MIDI_DIR, MIDI_THRES):
    # For each midi test file in the directory
    for root, directory, f in os.walk(MIDI_DIR):
        for midiName in f:
            if midiName[-4:].lower() == '.mid':
               src = os.path.join( root, midiName )
               # (midi_order_no for comparing with the ground truth)
               midi_order_no = midiName.split('-')[1]
               # get the codeword for the midi file
               MIDI_codewords = load_MIDI(src,MIDI_THRES) 
               # get a list of candidates from the codeIndex
               candidates = codeword_hashing.get_candidates(MIDI_codewords,codeIndex)
               # (This is for comparing with the ground truth file)
               for cand in candidates:
                   if cand[0].startswith(midi_order_no + '-'):
                      ct += 1
                      print ct
                      print midiName

