from apricot.functions.facilityLocation import FacilityLocationSelection
import time, pickle
import numpy as np


def facloc_anomaly(timestamps, timestamp_dims, mark):
    facloc = FacilityLocationSelection(int(0.7 * len(timestamps)))
    X = np.array([_ for _ in zip(timestamps, mark, timestamp_dims)])
    result = facloc.fit_transform(X)
    endo_time = np.array([r[0] for r in result])
    endo_mask = np.array([True if t in endo_time else False for t in timestamps])
    return np.argsort(endo_mask)


def facloc_pred(timestamps, timestamp_dims, mark, size):
    facloc = FacilityLocationSelection(size)
    X = np.array([_ for _ in zip(timestamps, mark, timestamp_dims)])
    result = facloc.fit_transform(X)
    endo_time = np.array([r[0] for r in result])
    endo_mask = np.array([True if t in endo_time else False for t in timestamps])
    left = np.where(endo_mask == False)[0]
    right = np.where(endo_mask != False)[0]
    np.random.shuffle(left)
    np.random.shuffle(right)
    return np.concatenate((left, right))


if __name__ == '__main__':

    filenames = ['BookOrder', 'Club']
    start = time.time()

    for filename in filenames:
        print(filename)
        with open('../../assets/data/%s.pickle' % filename, 'rb') as handle:
            timestamps, timestamp_dims, _, mark = pickle.load(handle)

        pred_exo_order = facloc_anomaly(timestamps, timestamp_dims, mark)

        with open('facloc_greedy_%s.pickle' % filename, 'wb') as handle:
            pickle.dump((pred_exo_order, None, None), handle, protocol=pickle.HIGHEST_PROTOCOL)
