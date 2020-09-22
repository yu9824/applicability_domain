import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from math import floor

# 例用
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def get_distance(XA, XB, sort = True):
    distance = cdist(XA, XB, metric = 'euclidean')    # ユークリッド距離を行ベクトルごとに算出する．distance.shape[0] == len(XA), distance.shape[1] == len(XB)
    if sort:
        distance.sort()
    return distance

def get_kNN_distance(XA, XB, k = 5, sort = True):
    distance_sorted = get_distance(XA, XB, sort = True)
    kNN_distance = np.mean(distance_sorted[:, :k+1], axis = 1)    # 周辺k個の点との距離の平均をkNN距離と定義
    if sort:
        kNN_distance.sort()
    arg = np.argsort(kNN_distance)  # 距離が近い順のindexをnp.ndarrayで返す
    return kNN_distance, arg

class ApplicabilityDomain:
    def __init__(self, k = 5, alpha = 0.95):
        self.k_in_kNN = k
        self.ratio = alpha

        '''
        scaler: MinMaxScaler() or StandardScaler()
        '''

    def get_ratio_distance(self, train, test, scaler = StandardScaler()):
        train = np.array(train)
        test = np.array(test)

        if train.shape[1] != test.shape[1]:
            raise ValueError("Either or Both of the inputs is/are not correct.")

        # それぞれ標準化 or 正規化
        train_normalized = scaler.fit_transform(train)
        test_normalized = scaler.transform(test)

        # train内でのkNN距離を求める．
        kNN_train_distance, arg_train = get_kNN_distance(train_normalized, train_normalized, k = self.k_in_kNN, sort = False)
        kNN_train_distance_sorted = np.sort(kNN_train_distance)

        # これより遠いと自信なくなるよ，の敷居を求める．
        threshold = kNN_train_distance_sorted[floor(kNN_train_distance_sorted.shape[0] * self.ratio) - 1]

        # trainとtestのkNN距離を算出
        kNN_train_test_distance, arg_train_test = get_kNN_distance(test_normalized, train_normalized, k = self.k_in_kNN, sort = False)

        # # 求めた敷居より近い位置にあるtestデータの番号 (index) をnp.ndarrayで取得
        # boolean = kNN_train_test_distance <= threshold

        # 敷居に対してどれくらいの割合なのか．（self.sample_numberを連続数化するための処理）
        # 小さければ小さいほど密度が高い→信頼度が高い．
        ratio_distance_train = kNN_train_distance / threshold
        ratio_distance_train_test = kNN_train_test_distance / threshold

        return ratio_distance_train, ratio_distance_train_test


    def is_inside(self, *args):
        return list(map(lambda x:x<=1, self.get_ratio_distance(*args)))






if __name__ == '__main__':
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

    AD = ApplicabilityDomain(k = 5, alpha = 0.95)
    boolean_train, boolean_test = AD.is_inside(X_train, X_test)
    print(X_train[boolean_train])
