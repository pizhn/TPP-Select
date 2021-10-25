import sys

sys.path.append(".")
import pickle

from predict.rmtpp_pred import rmtpp_predict

filename = ['BookOrder', 'Club', 'Election', 'Series', 'Verdict']

k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# filename method
if __name__ == '__main__':
    result = {}

    # rmtpp
    for _filename in filename:
        for _k in k:
            with open('./assets/real_data/%s.pickle' % filename, 'rb') as handle:
                timestamps, timestamp_dims, edge, mark = pickle.load(handle)
            with open('./assets/greedy_result/%s_greedy_result.pickle' % filename, 'rb') as handle:
                pred_exo_idxs = pickle.load(handle)

            # rmtpp
            rmtpp_time_error, rmtpp_mark_error = rmtpp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark, k=_k)
            result['rmtpp'][filename][_k]['time'] = rmtpp_time_error
            result['rmtpp'][filename][_k]['mark'] = rmtpp_mark_error

            # # thp
            # thp_time_error, thp_mark_error = thp_predict(pred_exo_idxs, timestamps, timestamp_dims, mark, k=_k)
            # result['thp'][filename][_k]['time'] = thp_time_error
            # result['thp'][filename][_k]['mark'] = thp_mark_error
