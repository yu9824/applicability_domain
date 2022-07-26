from typing import Optional, Union, List, Tuple
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.feature_selection import VarianceThreshold


class ApplicabilityDomainDetector(BaseEstimator, TransformerMixin):
    def __init__(self, k:int=5, alpha:float=0.997, p:int=2, algorithm:str='auto', n_jobs:Optional[int]=None):
        """Applicability Domain Detector.

        Parameters
        ----------
        k : int, optional
            k in kNN, by default 5
        alpha : float, optional
            , by default 0.95
        p : int, optional
            Minkowski distance parameter, by default 2
            when p==2, it mean euclidian distance.
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, by default 'auto'
            Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`BallTree`
            - 'kd_tree' will use :class:`KDTree`
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
            based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.
        n_jobs : Optional[int], optional, by default None
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.p = p
        self.algorithm = algorithm
        self.n_jobs = n_jobs
    
    def fit(self, X:Union[np.ndarray, pd.DataFrame], y=None)->ApplicabilityDomainDetector:
        # check
        self.X_:np.ndarray = check_array(X)
        vselector = VarianceThreshold(threshold=0)
        if np.sum(vselector.fit(self.X_).get_support()) != self.X_.shape[1]:
            raise ValueError('`X` must not contain features with variance zero.')

        X_scaled = self._scale(self.X_)
        self.nn = NearestNeighbors(n_neighbors=self.k, p=self.p, algorithm=self.algorithm, n_jobs=self.n_jobs)
        self.nn.fit(X_scaled)
        self.threshold_ = np.percentile(self.nn.kneighbors(X_scaled, return_distance=True)[0].mean(axis=1), self.alpha * 100)

        # calc on own data
        self.ratio_distance_ = self.get_ratio_distance(X)
        self.support_ = self.ratio_distance_ <= 1.0
        return self
    
    def _scale(self, X:Union[np.ndarray, pd.DataFrame])->np.ndarray:
        X:np.ndarray = check_array(X)
        return (X - self.X_.mean(axis=0)) / self.X_.std(ddof=1, axis=0)

    def transform(self, X:Union[np.ndarray, pd.DataFrame])->np.ndarray:
        check_is_fitted(self, 'threshold_')
        X:np.ndarray = check_array(X)
        return X[self.get_support(X)]
    
    def get_ratio_distance(self, X:Optional[Union[np.ndarray, pd.DataFrame]]=None)->np.ndarray:
        check_is_fitted(self, 'threshold_')
        if X is None:
            return self.ratio_distance_
        else:
            X:np.ndarray = check_array(X)

            X_scaled = self._scale(X)
            distances, indices = self.nn.kneighbors(X_scaled)
            return distances.mean(axis=1) / self.threshold_
    
    def get_support(self, X:Optional[Union[np.ndarray, pd.DataFrame]]=None)->np.ndarray:
        if X is None:
            return self.support_
        else:
            ratio_distance = self.get_ratio_distance(X)
            return ratio_distance <= 1.0


