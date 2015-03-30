'''
Functions for matching hash sequences quickly
'''
import numpy as np
import numba

N_BITS = 16

# Construct "bits-set-table"
bits_set = np.zeros(2**N_BITS, dtype=np.uint16)

for i in xrange(2**N_BITS):
    bits_set[i] = (i & 1) + bits_set[i/2]


def ints_to_vectors(int_sequence):
    '''
    Convert a sequence of integers into bit vector arrays

    :parameters:
        - int_sequence : np.ndarray, dtype=np.int
            Sequence of integers

    :returns:
        - vectors : np.ndarray, dtype=np.bool
            Matrix of bit vectors, shape (len(int_sequence), N_BITS)
    '''
    return np.array([[n >> i & 1 for i in range(N_BITS)]
                     for n in int_sequence])


def vectors_to_ints(vectors):
    '''
    Turn a matrix of bit vector arrays into a vector of ints

    :parameters:
        - vectors : np.ndarray
            Matrix of bit vectors, shape (n_vectors, n_bits)

    :returns:
        - ints : np.ndarray
            Vector of ints
    '''
    return (vectors*2**(np.arange(vectors.shape[1])*vectors)).sum(axis=1)


@numba.jit('void(u2[:], u2[:], u2[:, :], u2[:])',
           locals={'m': numba.uint16,
                   'n': numba.uint16},
           nopython=True)
def int_dist(x, y, output, bits_set=bits_set):
    '''
    Compute the pairwise bit-distance matrix of two sequences of integers.

    :parameters:
        - x : np.ndarray, dtype='uint16'
            Sequence of integers
        - y : np.ndarray, dtype='uint16'
            Sequence of integers
        - output : np.ndarray, dtype='uint16'
            Pre-allocated matrix where the pairwise distances will be stored.
            shape=(x.shape[0], y.shape[0])
        - bits_set : np.ndarray, dtype='uint16'
            Table where bits_set(x) is the number of 1s in the binary
            representation of x, where x is an unsigned 16 bit int
    '''
    nx = x.shape[0]
    ny = y.shape[0]
    # Populate the distance matrix
    for m in xrange(nx):
        for n in xrange(ny):
            # XORing ^ x[m] and y[n] will produce a 16-bit int where the i'th
            # bit is 1 when the i'th bit of x[m] and the i'th bit of y[n] are
            # the same.  Retrieving the entry in bits_set will then count
            # the number of entries in x[m] and y[n] which are the same.
            output[m, n] = bits_set[x[m] ^ y[n]]


@numba.jit('void(u2[:, :], u2, u2[:, :])',
           locals={'i': numba.uint16,
                   'j': numba.uint16},
           nopython=True)
def dtw_core(D, pen, path_length):
    '''
    Core dynamic programming routine for dynamic time warping.

    :parameters:
        - D : np.ndarray, dtype='uint16'
            Distance matrix
        - pen : int
            Non-diagonal move penalty
        - path_length : np.ndarray, dtype='uint16'
            Pre-allocated traceback matrix
    '''
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    for i in xrange(D.shape[0] - 1):
        for j in xrange(D.shape[1] - 1):
            # Diagonal move (which has no penalty) is lowest
            if D[i, j] <= D[i, j + 1] + pen and D[i, j] <= D[i + 1, j] + pen:
                path_length[i + 1, j + 1] += path_length[i, j] + 1
                D[i + 1, j + 1] += D[i, j]
            # Horizontal move (has penalty)
            elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] + pen <= D[i, j]:
                path_length[i + 1, j + 1] += path_length[i, j + 1] + 1
                D[i + 1, j + 1] += D[i, j + 1] + pen
            # Vertical move (has penalty)
            elif D[i + 1, j] <= D[i, j + 1] and D[i + 1, j] + pen <= D[i, j]:
                path_length[i + 1, j + 1] += path_length[i + 1, j] + 1
                D[i + 1, j + 1] += D[i + 1, j] + pen


def dtw(distance_matrix, gully, penalty):
    '''
    Compute the dynamic time warping distance between two sequences given a
    distance matrix.  The score is normalized by the path length.  Assumes an
    integer distance matrix.

    :parameters:
        - distance_matrix : np.ndarray, dtype='uint16'
            Distances between two sequences
        - gully : float
            Sequences must match up to this porportion of shorter sequence
        - penalty : int
            Non-diagonal move penalty

    :returns:
        - score : float
            DTW score of lowest cost path through the distance matrix.
    '''
    # Pre-allocate traceback matrix
    path_length = np.zeros(distance_matrix.shape, distance_matrix.dtype)
    # Populate distance matrix with lowest cost path
    dtw_core(distance_matrix, penalty, path_length)
    # Traceback from lowest-cost point on bottom or right edge
    gully = int(gully*min(distance_matrix.shape[0], distance_matrix.shape[1]))
    i = np.argmin(distance_matrix[gully:, -1]) + gully
    j = np.argmin(distance_matrix[-1, gully:]) + gully

    if distance_matrix[-1, j] > distance_matrix[i, -1]:
        j = distance_matrix.shape[1] - 1
    else:
        i = distance_matrix.shape[0] - 1

    # Score is the final score of the best path
    score = distance_matrix[i, j]/float(path_length[i, j])

    return score


def match_one_sequence(query, sequences, gully, penalty,
                       sequence_indices=None):
    '''
    Match a query sequence to one of the sequences in a list

    :parameters:
        - query : np.ndarray, dtype='uint16'
            Query sequence
        - sequences : list of np.ndarray, dtype='uint16'
            Sequences to find matches in, sorted by sequence length
        - gully : float
            Sequences must match up to this porportion of shorter sequence
        - penalty : int
            DTW Non-diagonal move penalty
        - sequence_indices : iterable or None
            Iterable of the indices of entries of `sequences` which should be
            matched against.  If `None`, match against all sequences.

    :returns:
        - matches : list of int
            List of match indices
        - scores : list of float
            Scores for each match
    '''
    # Pre-allocate match and score lists
    matches = []
    scores = []
    # Default: Check all sequences
    if sequence_indices is None:
        sequence_indices = xrange(len(sequences))
    for n in sequence_indices:
        # Compute distance matrix
        distance_matrix = np.empty(
            (query.shape[0], sequences[n].shape[0]), dtype=np.uint16)
        int_dist(query, sequences[n], distance_matrix, bits_set)
        # Compute DTW distance
        score = dtw(distance_matrix, gully, penalty)
        # Store the score/match (even if it's not the best)
        matches.append(n)
        scores.append(score)
    # Sort the scores and matches
    sorted_idx = np.argsort(scores)
    matches = [matches[n] for n in sorted_idx]
    scores = [scores[n] for n in sorted_idx]
    return matches, scores
