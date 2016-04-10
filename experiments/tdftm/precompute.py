"""
Precompute 2D FTM embeddings for all MSD entries and MIDI files from clean_midi
"""
# We'll use msgpack for I/O.  It seems to be fastest, and is widely supported.
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import deepdish
import collections
import glob
import os
import numpy as np
import librosa
import traceback
import sys
import scipy.fftpack
import sklearn.decomposition
sys.path.append(os.path.join('..', '..'))
import feature_extraction
import whoosh_search

RESULTS_PATH = '../../results'
DATA_PATH = '../../data'
# These are hyperparameters of the embedding process, chosen by Bertin-Mahieux
WINDOW = 75
POWER = 1.96
# We'll use a 128-dimensional reduction of the 2DFTM to match PSE
PCA_DIM = 128


def gram_to_beat_chroma(gram):
    '''
    Converts a pre-computed CQT to a beat-synchronous chromagram, transposed so
    that the first dimension are features and the second are time frames.
    This implements all that is needed to convert pre-computed CQTs to the
    format used in the 2DFTM experiments.

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram, shape=(n_frames, n_frequency_bins)

    Returns
    -------
    chroma : np.ndarray
        Beat-synchronous chroma matrix, shape (n_frequency_bins, n_beats)
    '''
    # Transpose to match librosa's format librosa
    gram = np.array(gram.T)
    # Because CQTs have spectra which are pre-L2-normalized, their range is
    # [-some number, 0]; this causes issues for the max-normalization which
    # happens below.  This rescales to [0, some_number]
    gram -= gram.min()
    # Compute beats
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=librosa.onset.onset_strength(S=gram),
        sr=feature_extraction.AUDIO_FS,
        hop_length=feature_extraction.AUDIO_HOP)
    # Make sure librosa didn't report 0 or 1 beats
    if beats.shape[0] < 2:
        # In this degenerate case, just put a beat at the beginning and the end
        # This, combined with the following interpolation, will result in an
        # even segmentation of the CQT into integrated frames
        beats = np.array([0, gram.shape[1]])
    # 2DFTM requires there to be at least 75 beats, so double the tempo until
    # there are 75 beats
    while beats.shape[0] < 75:
        # Linearly interpolate beats between all the existing beats
        interped_beats = np.empty(2 * beats.shape[0] - 1)
        interped_beats[::2] = beats
        interped_beats[1::2] = beats[:-1] + np.diff(beats) / 2.
        beats = interped_beats
    # Compute CQT from chroma, without any built-in normalization or threshold
    chroma = librosa.feature.chroma_cqt(
        C=gram, norm=None, threshold=None,
        fmin=librosa.midi_to_hz(feature_extraction.NOTE_START))
    # Compute beat-synchronous chroma
    beat_chroma = librosa.feature.sync(chroma, beats)
    # Max-normalize the result - this is done in Thierry/DAn's msd_beatchroma
    beat_chroma = librosa.util.normalize(beat_chroma)
    return beat_chroma


def chrompwr(X, P=POWER):
    """
    Raise entries of a matrix to a power, while preserving the column norm.
    Originally written in Matlab by Daniel P. W. Ellis, translated to Python by
    Thierry Bertin-Mahieux

    Parameters
    ----------
    X : np.ndarray
        Matrix to process.
    P : float
        Power to raise entries to

    Returns
    -------
    Y : np.ndarray
        Processed matrix.

    Note
    ----
    Original docstring:

    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[np.where(CMn == 0)] = 1
    # normalize each input col, raise to power
    CMp = np.power(X / CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn == 0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)


def btchroma_to_fftmat(btchroma, win=WINDOW):
    """
    Convert a beat-synchronous chroma matrix to a flattened 2D Fourier
    Transform Magnitude matrix.
    Written by Thierry Bertin-Mahieux

    Parameters
    ----------
    btchroma : np.ndarray
        Beat-synchronous chroma matrix, shape (n_frequency_bins, n_beats)

    win : int
        Number of beats to includea in each 2DFTM patch

    Returns
    -------
    fftmat : np.ndarray
        Flattened 2D Fourier Transform Magnitude matrix

    Note
    ----
    Original docstring:

    Stack the flattened result of fft2 on patches 12 x win
    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    """
    # 12 semitones
    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats - win + 1):
        patch = scipy.fftpack.fftshift(
            np.abs(scipy.fftpack.fft2(btchroma[:, i:i + win])))
        # 'F' to copy Matlab, otherwise 'C'
        fftmat[:, i] = patch.flatten('F')
    return fftmat


def compute_tdftm(gram):
    """
    Compute post-processed 2DFTM embedding of a CQT

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram of a song.

    Returns
    -------
    tdftm : np.ndarray
        Fixed-length embedding based on median 2DFTM over sliding windows.
    """
    # Compute beat-synchronous choma
    beat_chroma = gram_to_beat_chroma(gram)
    # Raise entries to a power
    beat_chroma = chrompwr(beat_chroma)
    # Construct 2DFTM
    tdftm = btchroma_to_fftmat(beat_chroma)
    # Compute median over patches
    tdftm = np.median(tdftm, axis=1)
    return tdftm


def post_process_tdftm(tdftm, pca_transform):
    """
    Use PCA to reduce dimensions of 2DFTM embedding and L2 normalize it

    Parameters
    ----------
    tdftm : np.ndarray
        2DFTM embedding vector.

    pca_transform : callable
        Function which PCA-transforms its argument.

    Returns
    -------
    tdftm : np.ndarray
        Dimensionality-reduced, L2 normalized 2DFTM embedding
    """
    # PCA (adding "n_samples" dimension)
    tdftm = pca_transform(tdftm.reshape(1, -1))
    # L2 Normalize
    return tdftm / np.sqrt(np.sum(tdftm**2))

if __name__ == '__main__':
    # Load in training data for computing mean for thresholding
    training_data = collections.defaultdict(list)
    training_glob = os.path.join(
        RESULTS_PATH, 'training_dataset_unaligned', 'train', 'h5', '*.h5')
    for f in glob.glob(training_glob):
        for k, v in deepdish.io.load(f).items():
            training_data[k].append(v)
    # Compute 2DFTM embeddings for each dataset/modality
    for dataset, network in zip(['clean_midi', 'msd'], ['X', 'Y']):
        # Get file list from whoosh index
        index = whoosh_search.get_whoosh_index(
            os.path.join(DATA_PATH, dataset, 'index'))
        with index.searcher() as searcher:
            file_list = list(searcher.documents())
        # We only need to hash MIDI files from the dev or test sets
        if dataset == 'clean_midi':
            md5s = [line.split(',')[0]
                    for pair_file in [
                        os.path.join(RESULTS_PATH, 'dev_pairs.csv'),
                        os.path.join(RESULTS_PATH, 'test_pairs.csv')]
                    for line in open(pair_file)
                    if line.split(',')[1] == 'msd']
            file_list = [e for e in file_list if e['id'] in md5s]
        # Construct 2DFTMs for the training set for this modality
        # Must index out the first (n_channels) dimension from each CQT
        training_tdftm = np.vstack(
            [compute_tdftm(d[0]) for d in training_data[network]])
        # Create PCA object for training set for this modality
        pca = sklearn.decomposition.PCA(PCA_DIM)
        pca.fit(training_tdftm)
        # Delete the training data to save memory
        del training_data[network]
        # Load in CQTs and write out downsampled hash sequences
        for entry in file_list:
            try:
                # Construct CQT h5 file path from file index entry
                h5_file = os.path.join(
                    DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                # Load in CQT
                gram = deepdish.io.load(h5_file)['gram']
                # Compute embedding for this sequence
                embedding = post_process_tdftm(
                    compute_tdftm(gram), pca.transform)
                # Construct output path to the same location in
                # RESULTS_PATH/dhs_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH, 'tdftm_{}_embeddings'.format(dataset),
                    entry['path'] + '.mpk')
                # Construct intermediate subdirectories if they don't exist
                if not os.path.exists(os.path.split(output_file)[0]):
                    os.makedirs(os.path.split(output_file)[0])
                # Save result, along with the index entry for convenience
                with open(output_file, 'wb') as f:
                    f.write(msgpack.packb(
                        dict(embedding=embedding, **entry)))
            except Exception:
                print "Error processing : {}".format(h5_file)
                print traceback.format_exc()
