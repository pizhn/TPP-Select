import sys, os, pickle
import numpy as np

from greedy.greedy import hybrid_greedy
from rmtpp_pred import rmtpp_predict
from func.common import train_test_ratio
from thp_pred import thp_predict

sys.path.append(".")

filename = ['BookOrder', 'Club', 'Election', 'Series', 'Verdict']

k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def get_endo_mask(exo_idxs, size):
    endo_mask = np.full(size, True)
    endo_mask[exo_idxs] = False
    return endo_mask


def select(timestamps, timestamp_dims, mark, n_iter, omega, v, penalty_time, penalty_mark, stochastic_size, skip_first):
    n_dim = len(np.unique(timestamp_dims))
    T = max(timestamps)
    sentiments = np.sort(np.unique(mark))
    size = len(timestamps)
    exo_idxs, _ = hybrid_greedy(timestamps, timestamp_dims, mark, n_iter, omega, v, n_dim, T,
                                penalty_time, penalty_mark, edge, sentiments,
                                stochastic_size=stochastic_size, verbose=True,
                                return_all=True, skip_first=skip_first)
    return get_endo_mask(exo_idxs, size)


# filename method
if __name__ == '__main__':
    result = {}
    n_trials = 5

    for _filename in filename:
        with open('./data/%s.pickle' % _filename, 'rb') as handle:
            timestamps, timestamp_dims, edge, mark = pickle.load(handle)

        for file in os.listdir('.'):
            if not file.startswith('%s_greedy_' % _filename):
                continue
            with open(os.path.join('.', file), 'rb') as handle:
                pred_exo_idxs = pickle.load(handle)

            for _k in k:
                # rmtpp
                rmtpp_time_error, rmtpp_mark_error = rmtpp_predict(pred_exo_idxs, timestamps, timestamp_dims,
                                                                   mark, train_test_ratio, k=_k)
                result['rmtpp'][_filename][_k]['time'] = rmtpp_time_error
                result['rmtpp'][_filename][_k]['mark'] = rmtpp_mark_error

                # # thp
                # _, _, _, _, _, thp_mark_error, thp_time_error = thp_predict(pred_exo_idxs, timestamps,
                #                                                             timestamp_dims, mark, k=_k)
                # result['thp', _filename, file, _k, 'time'] = thp_time_error
                # result['thp', _filename, file, _k, 'mark'] = thp_mark_error

                print(result)
