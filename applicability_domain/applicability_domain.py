'''
Copyright © 2021 yu9824
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist
from math import floor


class ApplicabilityDomainDetector(BaseEstimator, TransformerMixin):
    def __init__(self, k=5, alpha=0.95, scaler=StandardScaler()):
        """

        Parameters
        ----------
        k : int, optional
            `k` nearest neighbors, by default 5
        alpha : float, optional
            0~1. ratio of inlier sample, by default 0.95
        scaler : scikit-learn scaler instance like sklearn.preprocessing.StandardScaler(), optional
            , by default StandardScaler()
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.scaler = scaler

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : 
        y : You don't how to enter, optional
            , by default None

        Returns
        -------
        [type]
            [description]
        """
        # check
        X = check_array(X)

        # normalize
        self.train_normalized_ = self.scaler.fit_transform(X)

        # train内でのkNN距離を求める．
        self.kNN_train_distance_, self.arg_train_ = self._get_kNN_distance(self.train_normalized_, self.train_normalized_, k = self.k, sort = False)
        kNN_train_distance_sorted_ = np.sort(self.kNN_train_distance_)

        # これより遠いと自信なくなるよ，の敷居を求める．
        self.threshold_ = kNN_train_distance_sorted_[floor(kNN_train_distance_sorted_.shape[0] * self.alpha) - 1]
        return self

    def transform(self, X):
        """

        Parameters
        ----------
        X : 

        Returns
        -------
        np.ndarray (transformed X)
        """
        self.get_support(X)
        return X[self.support_]

    def get_support(self, X):
        """

        Parameters
        ----------
        X : 

        Returns
        -------
        np.ndarray (boolean)
        """
        self.get_ratio_distance(X)
        self.support_ = self.ratio_distance_ <= 1
        return self.support_

    def get_ratio_distance(self, X):
        """

        Parameters
        ----------
        X : 

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            If you enter an array with a different number of features, it will return an error.
        """
        # check_is_fitted
        check_is_fitted(self, 'threshold_')

        # check
        X = check_array(X)

        # normalize
        self.test_normalized_ = self.scaler.transform(X)

        # 
        if self.test_normalized_.shape[1] != self.train_normalized_.shape[1]:
            raise ValueError("Either or Both of the inputs is/are not correct.")

        # trainとtestのkNN距離を算出
        self.kNN_train_test_distance_, self.arg_train_test_ = self._get_kNN_distance(self.test_normalized_, self.train_normalized_, k = self.k, sort = False)

        # 敷居に対してどれくらいの割合なのか．（self.sample_numberを連続数化するための処理）
        # 小さければ小さいほど密度が高い→信頼度が高い．
        self.ratio_distance_ = self.kNN_train_test_distance_ / self.threshold_
        return self.ratio_distance_


    def _get_distance(self, XA, XB, sort = True):
        distance = cdist(XA, XB, metric = 'euclidean')    # ユークリッド距離を行ベクトルごとに算出する．distance.shape[0] == len(XA), distance.shape[1] == len(XB)
        if sort:
            distance.sort()
        return distance

    def _get_kNN_distance(self, XA, XB, k = 5, sort = True):
        distance_sorted = self._get_distance(XA, XB, sort = True)
        kNN_distance = np.mean(distance_sorted[:, :k+1], axis = 1)    # 周辺k個の点との距離の平均をkNN距離と定義
        if sort:
            kNN_distance.sort()
        arg = np.argsort(kNN_distance)  # 距離が近い順のindexをnp.ndarrayで返す
        return kNN_distance, arg


if __name__ == '__main__':
    pass
