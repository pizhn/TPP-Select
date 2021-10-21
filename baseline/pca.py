import numpy as np
from sklearn.decomposition import PCA
import pickle
import time


def deviation(x, eigenvalues, feature_vectors):
    d = 0.0
    for eigenvalue, feature_vector in zip(eigenvalues, feature_vectors):
        d += np.sum(np.inner(x, feature_vector)**2) / eigenvalue
    return d


def PCA_anomaly(timestamp, timestamp_dims, mark, size, k=3):
    pca = PCA(n_components=k)
    X = [_ for _ in zip(timestamp, mark, timestamp_dims)]
    pca.fit(X)
    devs = [deviation(x, pca.explained_variance_, pca.components_) for x in X]
    pred_exo_order = np.argsort(devs)[::-1]
    return pred_exo_order


def pca_pred(timestamp, timestamp_dims, mark, size, k=3):
    pca = PCA(n_components=k)
    X = [_ for _ in zip(timestamp, mark, timestamp_dims)]
    pca.fit(X)
    devs = [deviation(x, pca.explained_variance_, pca.components_) for x in X]
    pred_exo_order = np.argsort(devs)[::-1]
    return pred_exo_order[:size]


if __name__ == '__main__':

    filenames = ['BookOrder', 'Club']
    start = time.time()

    for filename in filenames:
        print(filename)
        with open('../../assets/data/%s.pickle' % filename, 'rb') as handle:
            timestamps, timestamp_dims, _, mark = pickle.load(handle)

        pred_exo_order = PCA_anomaly(timestamps, timestamp_dims, mark)

        with open('PCA_greedy_%s.pickle' % filename, 'wb') as handle:
            pickle.dump((pred_exo_order, None, None), handle, protocol=pickle.HIGHEST_PROTOCOL)
