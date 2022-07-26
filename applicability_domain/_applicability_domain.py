from __future__ import annotations
from typing import Optional, Union, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import OutlierMixin
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.utils.metaestimators import available_if
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


class ApplicabilityDomain(KNeighborsMixin, OutlierMixin, NeighborsBase):
    def __init__(
        self,
        n_neighbors=20,
        alpha:float=0.997,
        *,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        novelty=False,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.novelty = novelty
        self.alpha = alpha

    def _check_novelty_fit_predict(self):
        if self.novelty:
            msg = (
                "fit_predict is not available when novelty=True. Use "
                "novelty=False if you want to predict on the training set."
            )
            raise AttributeError(msg)
        return True
    
    @available_if(_check_novelty_fit_predict)
    def fit_predict(self, X, y=None):
        """Fit the model to the training set X and return the labels.

        **Not available for novelty detection (when novelty is set to True).**
        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        # As fit_predict would be different from fit.predict, fit_predict is
        # only available for outlier detection (novelty=False)

        return self.fit(X)._predict()

    def fit(self, X, y=None):
        """Fit the local outlier factor detector from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted local outlier factor detector.
        """
        self._fit(X)

        n_samples = self.n_samples_fit_
        if self.n_neighbors > n_samples:
            warnings.warn(
                "n_neighbors (%s) is greater than the "
                "total number of samples (%s). n_neighbors "
                "will be set to (n_samples - 1) for estimation."
                % (self.n_neighbors, n_samples)
            )
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_
        )

        self.threshold_ = np.percentile(self._distances_fit_X_.mean(axis=1), self.alpha * 100)

        # calc on own data
        self.ratio_distance_ = self.get_ratio_distance(X)
        self.support_ = self.ratio_distance_ <= 1.0
        return self

    def get_ratio_distance(self, X:Optional[Union[np.ndarray, pd.DataFrame]]=None)->np.ndarray:
        check_is_fitted(self, 'threshold_')
        if X is None:
            return self.ratio_distance_
        else:
            X:np.ndarray = check_array(X)
            distances, indices = self.kneighbors(X)
            return distances.mean(axis=1) / self.threshold_

    def _check_novelty_predict(self):
        if not self.novelty:
            msg = (
                "predict is not available when novelty=False, use "
                "fit_predict if you want to predict on training data. Use "
                "novelty=True if you want to use LOF for novelty detection "
                "and predict on new unseen data."
            )
            raise AttributeError(msg)
        return True
    
    @available_if(_check_novelty_predict)
    def predict(self, X=None):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        **Only available for novelty detection (when novelty is set to True).**
        This method allows to generalize prediction to *new observations* (not
        in the training set). Note that the result of ``clf.fit(X)`` then
        ``clf.predict(X)`` with ``novelty=True`` may differ from the result
        obtained by ``clf.fit_predict(X)`` with ``novelty=False``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        return self._predict(X)

    def _predict(self, X=None):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        If X is None, returns the same as fit_predict(X_train).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples. If None, makes prediction on the
            training data without considering them as their own neighbors.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        check_is_fitted(self)

        if X is not None:
            X:np.ndarray = check_array(X, accept_sparse="csr")
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self.get_ratio_distance(X) > self.threshold_] = -1
        else:
            is_inlier = np.ones(self.n_samples_fit_, dtype=int)
            is_inlier[~self.support_] = -1

        return is_inlier
    
    def _check_novelty_decision_function(self):
        if not self.novelty:
            msg = (
                "decision_function is not available when novelty=False. "
                "Use novelty=True if you want to use LOF for novelty "
                "detection and compute decision_function for new unseen "
                "data. Note that the opposite LOF of the training samples "
                "is always available by considering the "
                "negative_outlier_factor_ attribute."
            )
            raise AttributeError(msg)
        return True
    
    def _check_novelty_score_samples(self):
        if not self.novelty:
            msg = (
                "score_samples is not available when novelty=False. The "
                "scores of the training samples are always available "
                "through the negative_outlier_factor_ attribute. Use "
                "novelty=True if you want to use LOF for novelty detection "
                "and compute score_samples for new unseen data."
            )
            raise AttributeError(msg)
        return True
    

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import LocalOutlierFactor
    
    iris = load_iris(as_frame=True)
    X:pd.DataFrame = iris.data
    y:pd.Series = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(ddof=1, axis=0)
    X_test_scaled = (X_test - X_train.mean(axis=0)) / X_train.std(ddof=1, axis=0)
    ad = ApplicabilityDomain(n_neighbors=5, alpha=0.95, novelty=True)
    ad.fit(X_train_scaled)
    print(ad.predict(X_test_scaled))
    
    # lof = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(X_train_scaled)
    # print(lof.predict(X_test_scaled))