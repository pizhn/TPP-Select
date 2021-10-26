from func.generate_synthetic import exo_by_num
import pickle, time
import numpy as np
from greedy.greedy import hybrid_greedy
from params import params_classification

if __name__ == '__main__':
    # filename = ['BookOrder', 'Club', 'Election', 'Series', 'Verdict']
    filenames = ['Series']

    n_trial = 3

    for i in range(n_trial):
        start = time.time()
        for filename in filenames:
            params = params_classification[filename]
            skip_first = params['skip_first']
            stochastic_size = params['stochastic_size']

            with open('./data/%s.pickle' % filename, 'rb') as handle:
                timestamps, timestamp_dims, edge, mark = pickle.load(handle)

            penalty_time, penalty_mark, omega, v = params['penalty_time'], params['penalty_mark'], \
                                                   params['omega'], params['v']
            skip_first = params['skip_first']
            stochastic_size = params['stochastic_size']

            pred_exo_idxs_ours, _ = hybrid_greedy(timestamps, timestamp_dims, mark, len(timestamps),
                                                  omega, v, len(np.unique(timestamp_dims)), max(timestamps),
                                                  penalty_time, penalty_mark, edge, np.sort(np.unique(mark)),
                                                  stochastic_size=stochastic_size, verbose=True,
                                                  return_all=True, skip_first=skip_first)

            with open('%s_greedy_%s.pickle' % (filename, start), 'wb') as handle:
                pickle.dump(pred_exo_idxs_ours, handle, protocol=pickle.HIGHEST_PROTOCOL)
