# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import codeword_hashing
import os

# <codecell>

MSD_DIR = '/Volumes/MILLIONSONG/data'
MSD_SUB_DIR = '/MillionSongSubset/data'
MIDI_DIR = '/midi_files_to_test/'

THRES_MSD = 0.15
THRES_MIDI = 0

# <codecell>

MSD_Hashing = codeword_hashing.load_MSD(MSD_SUB_DIR, THRES_MSD)

# <codecell>

for root, directory, f in os.walk(MIDI_DIR):
    for midiName in f:
        if midiName[-4:].lower() == '.mid':
           #print midiName
           src = os.path.join( root, midiName )
           # Get the order number for the midi file from the name
           number = midiName.split('-')[1]
           # Getting the midi codeword
           MIDI_codeword = codeword_hashing.load_MIDI(src,0) 
           # Using the hashing to find a list of songs that contain the midi codeword
           candidates = codeword_hashing.find_song_id(MIDI_codeword,MSD_Hashing)
           #cand_ct = 0
           for cand in candidates:
               # Selecting top 150 cancdidates
               #cand_ct += 1
               #if cand[0].startswith(number + '-') and cand_ct <= 150:
               if cand[0].startswith(number + '-'):
                  ct += 1
                  print ct
                  print midiName

