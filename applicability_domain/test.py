if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from applicability_domain import ApplicabilityDomainDetector

    # サンプルデータ
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

    ad = ApplicabilityDomainDetector()
    ad.fit(X_train)
    print('fit is OK.')

    del ad
    ad = ApplicabilityDomainDetector()
    ad.fit_transform(X_train)

    ad.transform(X_test)

    support_train = ad.get_support(X_train)
    support_test = ad.get_support(X_test)

    X_train[support_train]
    X_test[support_test]

    print('success!')