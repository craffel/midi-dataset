# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/usr/bin/env python
# CREATED:2013-07-10 10:14:41 by Brian McFee <brm2132@columbia.edu>
# wrapper for sklearn estimators to buffer generator output for use with stochastic
# optimization via partial_fit()

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BufferedEstimator(BaseEstimator):

    def __init__(self, estimator, batch_size=256, sparse=False):
        """
        :parameters:
            - estimator : sklearn.BaseEstimator
                Any classifier/transformer that supports a partial_fit method

            - batch_size : int > 0
                The amount of data to buffer for each call to partial_fit

            - sparse : boolean
                Is the data generator sparse?
        """

        # Make sure that the estimator is of the right base type
        if not isinstance(estimator, BaseEstimator):
            raise TypeError('estimator must extend from sklearn.base.BaseEstimator')

        self.estimator  = estimator
        self.batch_size = batch_size
        self.sparse     = sparse

        #  are we classifying or transforming?
        self.supervised = isinstance(estimator, ClassifierMixin)

        #  this will only work if the estimator supports partial_fit
        assert hasattr(estimator, 'partial_fit')


    def fit(self, generator):

        def _run(X):
            if self.supervised:
                y = np.array([z[-1] for z in X])
                X = np.array([z[0] for z in X])
                self.estimator.partial_fit(X, y)
            else:
                X = np.array(X)
                self.estimator.partial_fit(X)

        X = []
        for (i, x_new) in enumerate(generator):
            X.append(x_new)
            if len(X) == self.batch_size:
                _run(X)
                X = []

        # Fit the last batch, if there is one
        if len(X) > 0:
            _run(X)

