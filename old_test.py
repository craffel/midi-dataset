# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import msd_matching
import numpy as np
import midi
import midi_to_features
import beat_aligned_feats
import os
#import codebook_training

from VectorQuantizer import VectorQuantizer
from sklearn.cluster import MiniBatchKMeans
import librosa
import glob
from datastream import datastream
from BufferedEstimator import BufferedEstimator

# <codecell>

# Training the million song subset to get a codebook
myDir = '/Users/rockzongli/Desktop/labRosa/MillionSongSubset/'
mySize = np.power(2,12)
codebook_training.train(myDir,mySize,'est4096')

# <codecell>

# Load the estimator and plot the codebook
f = open('estimator4096.p','rb')
myEstimator = pickle.load(f)
f.close()
myCodebook = myEstimator.components_.T
plt.imshow(myCodebook, aspect='auto', interpolation='none', origin='lower')

# <codecell>

cwd = os.getcwd()
myDir = cwd + '/data/'
testData = midi.read_midifile(myDir+'2414Letitbe.mid')
noteMatrix, beats, fs = midi_to_features.get_onsets_and_notes(testData)
beatChroma = midi_to_features.get_beat_chroma(noteMatrix,beats)
midi_beatChroma = midi_to_features.get_normalize_beatChroma(beatChroma)
plt.imshow(midi_beatChroma, origin='lower', aspect='auto', interpolation='nearest' )

# <codecell>

midi_cw = myEstimator.transform(midi_beatChroma.T).todense()
midi_codeword = np.nonzero(midi_cw)[1]
plot(np.array (midi_cw ).sum(axis=0))

# <codecell>

msd_beatChroma = beat_aligned_feats.get_btchromas(os.getcwd() + '/data/TRFBNQN128F92FFA5E(letitbe).h5')
#    if btChroma is not None and not np.isnan(btChroma).any()
msd_cw = myEstimator.transform(msd_beatChroma.T).todense()
msd_codeword = np.nonzero(msd_cw)[1]
plot(np.array (msd_cw ).sum(axis=0))

