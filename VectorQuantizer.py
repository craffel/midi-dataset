# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/usr/bin/env python
# CREATED:2013-05-05 11:14:34 by Brian McFee <brm2132@columbia.edu>
# sklearn.decomposition container class for vector quantization 

import numpy as np
import scipy.sparse
import sklearn.cluster
from sklearn.base import BaseEstimator, TransformerMixin

class VectorQuantizer(BaseEstimator, TransformerMixin):

    def __init__(self, clusterer=None, n_atoms=32, sparse=True, batch_size=1024):
        '''Vector quantization by closest centroid:

        A[i] == 1 <=> i = argmin ||X - C_i||
                        i

        Arguments:
        ----------
        n_atoms : int
            Number of dictionary elements to extract
    
        clusterer : {None, BaseEstimator}
            Instantiation of a clustering object (eg. sklearn.cluster.MiniBatchKMeans)

            default: sklearn.cluster.MiniBatchKMeans

        n_atoms : int
            If no clusterer is provided, the number of atoms to use

        sparse : bool
            Represent encoded data as a sparse matrix or ndarray

        batch_size : int
            Number of points to transform in parallel
        '''

        if clusterer is None:
            self.clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=n_atoms)
        else:
            self.clusterer = clusterer

        self.sparse = sparse
        self.batch_size = batch_size


    def fit(self, X):
        '''Fit the codebook to the data

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data

        Returns
        -------
        self : object
        '''
        
        self.clusterer.fit(X)
        self.center_norms_ = 0.5 * (self.clusterer.cluster_centers_**2).sum(axis=1)
        self.components_ = self.clusterer.cluster_centers_

        return self

    def partial_fit(self, X):
        self.clusterer.partial_fit(X)
        self.center_norms_ = 0.5 * (self.clusterer.cluster_centers_**2).sum(axis=1)
        self.components_ = self.clusterer.cluster_centers_

        return self


    def transform(self, X):
        '''Encode the data by VQ.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
        '''

        C = self.clusterer.cluster_centers_

        n = X.shape[0]

        hits = np.empty(n, dtype=np.uint16)

        for j in range(0, n, self.batch_size):
            j_end = min(n, j + self.batch_size)

            XC = np.dot(X[j:j_end], C.T) - self.center_norms_
            hits[j:j_end] = XC.argmax(axis=1)

        if self.sparse:
            X_new = scipy.sparse.lil_matrix( (n, C.shape[0]))
        else:
            X_new = np.zeros( (n, C.shape[0]), dtype=bool )
        
        for i in range(n):
            X_new[i, hits[i]] = True

        if self.sparse:
            X_new = X_new.tocsc()

        return X_new

