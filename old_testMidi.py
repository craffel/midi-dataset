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
testData = midi.read_midifile("2414Letitbe.mid")
noteMatrix, beats, fs = midi_to_features.get_onsets_and_notes(testData)
beatChroma = midi_to_features.get_beat_chroma(noteMatrix,beats)
midi_beatChroma = midi_to_features.get_normalize_beatChroma(beatChroma)
plt.imshow(midi_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

# Getting beat-synchronized Chromagram from MSD file
msd_beatChroma = beat_aligned_feats.get_btchromas('TRFBNQN128F92FFA5E(letitbe).h5')
plt.imshow(msd_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

# Vector Quantization 
thres = 0.4
msd_beatChroma[msd_beatChroma < thres] = 0
msd_beatChroma[msd_beatChroma >= thres] = 1
plt.imshow(msd_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

midiThres = 0.1

midi_beatChroma[midi_beatChroma < midiThres] = 0
midi_beatChroma[midi_beatChroma >= midiThres] = 1
plt.imshow(midi_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

# Create chroma-code for midi

midi_hash = np.zeros(midi_beatChroma.shape[1])
msd_hash = np.zeros(msd_beatChroma.shape[1])
for col in range(midi_hash.size):
    for row in range(12):
        midi_hash[col] += np.power(2*midi_beatChroma[row][col],11-row)

for col in range(midi_hash.size):
    for row in range(12):
        msd_hash[col] += np.power(2*msd_beatChroma[row][col],11-row)

plot(msd_hash)
plot(midi_hash,'r')

# <codecell>

print msd_hash
print midi_hash
plot(midi_hash)

