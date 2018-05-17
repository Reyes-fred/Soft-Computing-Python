import numpy as np
import scipy.sparse
import sklearn.datasets
import sklearn.ensemble


def get_dataset(dataset='boston', make_sparse=False):
    iris = getattr(sklearn.datasets, "load_%s" % dataset)()
    X = iris.data.astype(np.float32)
    Y = iris.target
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    train_size = int(len(indices) / 3. * 2.)
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]

    if make_sparse:
        X_train[:, 0] = 0
        X_train[rs.random_sample(X_train.shape) > 0.5] = 0
        X_train = scipy.sparse.csc_matrix(X_train)
        X_train.eliminate_zeros()
        X_test[:, 0] = 0
        X_test[rs.random_sample(X_test.shape) > 0.5] = 0
        X_test = scipy.sparse.csc_matrix(X_test)
        X_test.eliminate_zeros()

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, _, _ = get_dataset(dataset='boston')
print(type(X_train))

classifier = sklearn.ensemble.GradientBoostingRegressor(warm_start=True)
classifier.fit(X_train, Y_train)
classifier.n_estimators += 1
classifier.fit(X_train, Y_train)
print('Fitted dense data')

X_train, Y_train, _, _ = get_dataset(dataset='boston', make_sparse=True)
print(type(X_train))

classifier = sklearn.ensemble.GradientBoostingRegressor(warm_start=True)
classifier.fit(X_train, Y_train)
classifier.n_estimators += 1
classifier.fit(X_train, Y_train)
print('Fitted sparse data')
