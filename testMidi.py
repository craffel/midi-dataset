# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import midi
import midi_to_features
import librosa
import beat_aligned_feats

# <codecell>


# Getting beat-synchronized chromagram from midi file
testData = midi.read_midifile("3565Hero.mid")
noteMatrix, bpm, beats, fs = midi_to_features.get_onsets_and_notes(testData)

beatChroma = midi_to_features.get_beat_chroma(noteMatrix,beats)
midi_beatChroma = midi_to_features.get_normalize_beatChroma(beatChroma)

plt.imshow(midi_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

# Getting beat-synchronized Chromagram from MSD file
msd_beatChroma1 = beat_aligned_feats.get_btchromas('TRGFCEO128F9342EC9.h5')
msd_beatChroma2 = beat_aligned_feats.get_btchromas('TRJWCEA128F4273174(hero).h5')
plt.imshow(msd_beatChroma2, origin='lower', aspect='auto', interpolation='nearest' )

