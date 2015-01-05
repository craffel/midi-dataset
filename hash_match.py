'''
Functions for matching hash sequences quickly
'''
import numpy as np


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
