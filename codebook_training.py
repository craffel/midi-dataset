# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import midi
import midi_to_features
import beat_aligned_feats

from VectorQuantizer import VectorQuantizer
from sklearn.cluster import MiniBatchKMeans
import librosa
import glob
from datastream import datastream
from BufferedEstimator import BufferedEstimator
import pickle

# <codecell>

# Map a filename into beat chroma, and yield column by column
def mapper(filename, n=20):
    """beat Chroma extractor.
       Given a .h5 filename, yields bt-chroma col by col
    """
    btChroma = beat_aligned_feats.get_btchromas(filename)
    if btChroma is not None and not np.isnan(btChroma).any():
        #print filename
        for b in btChroma.T:
            yield b

# <codecell>

def train(myDir, mySize, bookName):
    """ train the million song subset to get a codebook
        myDir - the full path of the million song subset
                i.e: /Users/kittyshi/labRosa/MillionSongSubset/
        size  - size of the codebook, numbers of different codewords
                i.e: np.power(2,12)
        bookName - name of the .npy 
        (Takes about 20 - 30 minutes)
    """
    
    fullPath = myDir + 'data/*/*/*/*.h5'
    files = sorted(glob.glob(fullPath))
    
    # Using Brian Mcfee's VectorQuantizer
    # From https://github.com/bmcfee/ml_scraps
    data_generator = datastream(mapper, files, k =16)
    estimator = VectorQuantizer(n_atoms = mySize)
    buf_est = BufferedEstimator(estimator, batch_size=mySize)
    buf_est.fit(data_generator)
    
    # Save the codebook (estimator)
    # codebook = estimator.components_.T
    np.save(bookName,codebook)
    f = open(bookName,'wb')
    pickle.dump(estimator,f)
    f.close()
    

