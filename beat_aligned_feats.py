"""
Thierry Bertin-Mahieux (2011) Columbia University
tb2332@columbia.edu

Code to get beat-aligned features (chromas or timbre)
from the HDF5 song files of the Million Song Dataset.

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.


Copyright 2011, Thierry Bertin-Mahieux
parts of this code from Ron J. Weiss

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import time
import glob
import numpy as np
try:
    import hdf5_getters as GETTERS
except ImportError:
    print 'cannot find file hdf5_getters.py'
    print 'you must put MSongsDB/PythonSrc in your path or import it otherwise'
    raise


def get_btchromas(h5):
    """
    Get beat-aligned chroma from a song file of the Million Song Dataset
    INPUT:
       h5          - filename or open h5 file
    RETURN:
       btchromas   - beat-aligned chromas, one beat per column
                     or None if something went wrong (e.g. no beats)
    """
    # if string, open and get chromas, if h5, get chromas
    if type(h5).__name__ == 'str':
        h5 = GETTERS.open_h5_file_read(h5)
        chromas = GETTERS.get_segments_pitches(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
        h5.close()
    else:
        chromas = GETTERS.get_segments_pitches(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
    # get the series of starts for segments and beats
    # NOTE: MAYBE USELESS?
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstarts.shape = (708,)
    #    btstarts.shape = (304,)
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # aligned features
    btchroma = align_feats(chromas.T, segstarts, btstarts, duration)
    if btchroma is None:
        return None
    # Renormalize. Each column max is 1.
    maxs = btchroma.max(axis=0)
    maxs[np.where(maxs == 0)] = 1.
    btchroma = (btchroma / maxs)
    # done
    return btchroma


def get_btchromas_loudness(h5):
    """
    Similar to btchroma, but adds the loudness back.
    We use the segments_loudness_max
    There is no max value constraint, simply no negative values.
    """
    # if string, open and get chromas, if h5, get chromas
    if type(h5).__name__ == 'str':
        h5 = GETTERS.open_h5_file_read(h5)
        chromas = GETTERS.get_segments_pitches(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
        loudnessmax = GETTERS.get_segments_loudness_max(h5)
        h5.close()
    else:
        chromas = GETTERS.get_segments_pitches(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
        loudnessmax = GETTERS.get_segments_loudness_max(h5)
    # get the series of starts for segments and beats
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # add back loudness
    chromas = chromas.T * idB(loudnessmax)
    # aligned features
    btchroma = align_feats(chromas, segstarts, btstarts, duration)
    if btchroma is None:
        return None
    # done (no renormalization)
    return btchroma


def get_bttimbre(h5):
    """
    Get beat-aligned timbre from a song file of the Million Song Dataset
    INPUT:
       h5          - filename or open h5 file
    RETURN:
       bttimbre    - beat-aligned timbre, one beat per column
                     or None if something went wrong (e.g. no beats)
    """
    # if string, open and get timbre, if h5, get timbre
    if type(h5).__name__ == 'str':
        h5 = GETTERS.open_h5_file_read(h5)
        timbre = GETTERS.get_segments_timbre(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
        h5.close()
    else:
        timbre = GETTERS.get_segments_timbre(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
    # get the series of starts for segments and beats
    # NOTE: MAYBE USELESS?
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstarts.shape = (708,)
    #    btstarts.shape = (304,)
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # aligned features
    bttimbre = align_feats(timbre.T, segstarts, btstarts, duration)
    if bttimbre is None:
        return None
    # done (no renormalization)
    return bttimbre


def get_btloudnessmax(h5):
    """
    Get beat-aligned loudness max from a song file of the Million Song Dataset
    INPUT:
       h5             - filename or open h5 file
    RETURN:
       btloudnessmax  - beat-aligned loudness max, one beat per column
                        or None if something went wrong (e.g. no beats)
    """
    # if string, open and get max loudness, if h5, get max loudness
    if type(h5).__name__ == 'str':
        h5 = GETTERS.open_h5_file_read(h5)
        loudnessmax = GETTERS.get_segments_loudness_max(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
        h5.close()
    else:
        loudnessmax = GETTERS.get_segments_loudness_max(h5)
        segstarts = GETTERS.get_segments_start(h5)
        btstarts = GETTERS.get_beats_start(h5)
        duration = GETTERS.get_duration(h5)
    # get the series of starts for segments and beats
    # NOTE: MAYBE USELESS?
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstarts.shape = (708,)
    #    btstarts.shape = (304,)
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # reverse dB
    loudnessmax = idB(loudnessmax)
    # aligned features
    btloudnessmax = align_feats(loudnessmax.reshape(1,
                                                    loudnessmax.shape[0]),
                                segstarts, btstarts, duration)
    if btloudnessmax is None:
        return None
    # set it back to dB
    btloudnessmax = dB(btloudnessmax + 1e-10)
    # done (no renormalization)
    return btloudnessmax


def align_feats(values, times, new_times, duration):
    '''
    Perform soft mean aggregation of `values` over the intervals in `new_times`
    If any of `new_times` falls outside of the limit of `times`,
    the edge values are used.

    Parameters
    ----------

    values : np.ndarray [shape=(n_features, n_time_steps - 1)]
        `values[n]` is the feature vector from `times[n]` to `times[n + 1]`

    times : np.ndarray [shape=(n_time_steps,)]
        Time boundaries of the original values

    new_times : np.ndarray [shape=(n_new_time_steps,)]
        Intervals over which to aggregate `values`

    Returns
    -------

    new_values : np.ndarray [shape=(n_features, n_new_time_steps - 1)]
        `values` aggregated over `new_times`
    '''
    times = np.append(times, duration)
    new_times = np.append(new_times, duration)
    # Adjust times to span new_times, effectively using edge values
    if new_times[0] < times[0]:
        times[0] = new_times[0]
    if new_times[-1] > times[-1]:
        times[-1] = new_times[-1]
    # Find the first time after each new_time
    first_after = np.argmax(np.less.outer(new_times, times), axis=1)
    # Find the first time before each new_time
    last_before = first_after - 1
    # Add the new times to the list of times
    times = np.append(times[:-1], new_times)
    # Add the values at the new times into values
    values = np.hstack([values, values[:, last_before]])
    # Re-sort times with the new times
    time_sort = np.argsort(times)
    times = times[time_sort]
    # apply the same sorting to the values
    values = values[:, time_sort]
    # Find the indices of times equal to the new times
    equal_idx = np.argmax(np.equal.outer(new_times, times), axis=1)
    # Compute interval lengths
    lengths = np.diff(times)
    new_lengths = np.diff(new_times)
    # Aggregate over all new time intervals
    return np.array([(lengths[a:b]*values[:, a:b]/new_lengths[n]).sum(axis=1)
                     for n, (a, b)
                     in enumerate(zip(equal_idx[:-1], equal_idx[1:]))]).T


def idB(loudness_array):
    """
    Reverse the Echo Nest loudness dB features.
    'loudness_array' can be pretty any numpy object:
    one value or an array
    Inspired by D. Ellis MATLAB code
    """
    return np.power(10., loudness_array / 20.)


def dB(inv_loudness_array):
    """
    Put loudness back in dB
    """
    return np.log10(inv_loudness_array) * 20.


def die_with_usage():
    """ HELP MENU """
    print 'beat_aligned_feats.py'
    print '   by T. Bertin-Mahieux (2011) Columbia University'
    print '      tb2332@columbia.edu'
    print ''
    print 'This code is intended to be used as a library.'
    print 'For debugging purposes, you can launch:'
    print '   python beat_aligned_feats.py <SONG FILENAME>'
    sys.exit(0)


if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    print '*** DEBUGGING ***'

    # params
    h5file = sys.argv[1]
    if not os.path.isfile(h5file):
        print 'ERROR: file %s does not exist.' % h5file
        sys.exit(0)

    # compute beat chromas
    btchromas = get_btchromas(h5file)
    print 'btchromas.shape =', btchromas.shape
    if np.isnan(btchromas).any():
        print 'btchromas have NaN'
    else:
        print 'btchromas have no NaN'
    print 'the max value is:', btchromas.max()

    # compute beat timbre
    bttimbre = get_bttimbre(h5file)
    print 'bttimbre.shape =', bttimbre.shape
    if np.isnan(bttimbre).any():
        print 'bttimbre have NaN'
    else:
        print 'bttimbre have no NaN'
    print 'the max value is:', bttimbre.max()
