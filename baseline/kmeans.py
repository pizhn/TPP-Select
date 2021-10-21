import pickle

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def Kmeans_anomaly(timestamps, timestamp_dims, mark):
    X = [_ for _ in zip(timestamps, mark, timestamp_dims)]
    kmeans = KMeans(n_clusters=2).fit(X)
    kmeans_y = kmeans.predict(X)
    return kmeans_y


def kmeans_pred(timestamps, timestamp_dims, mark, size):
    X = [_ for _ in zip(timestamps, mark, timestamp_dims)]
    kmeans = KMeans(n_clusters=2).fit(X)
    kmeans_y = kmeans.predict(X)
    left = np.where(kmeans_y == 1)[0]
    right = np.where(kmeans_y != 1)[0]
    np.random.shuffle(left)
    np.random.shuffle(right)
    return np.concatenate((left, right))[:size]

if __name__ == '__main__':

    filenames = ['BookOrder', 'Club']

    for filename in filenames:
        with open('../../assets/data/%s.pickle' % filename, 'rb') as handle:
            timestamps, timestamp_dims, _, mark = pickle.load(handle)

        kmeans_y = Kmeans_anomaly(timestamps, timestamp_dims, mark)

        with open('kmeans_greedy_%s.pickle' % filename, 'wb') as handle:
            pickle.dump((kmeans_y, None, None), handle, protocol=pickle.HIGHEST_PROTOCOL)