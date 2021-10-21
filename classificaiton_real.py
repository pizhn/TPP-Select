from baseline.pca import pca_pred
from baseline.em import em_pred
from baseline.facloc import facloc_pred
from baseline.kmeans import kmeans_pred
from func.generate_synthetic import exo_by_num
import pickle
import numpy as np
import itertools
from greedy.greedy import hybrid_greedy

if __name__ == '__main__':
    filenames = ['BookOrder', 'Club', 'Election', 'Series', 'Verdict']
    stochastic_size = 100
    iter = 50
    accus_ours = {}
    accus_facloc = {}
    accus_kmeans = {}
    accus_pca = {}
    accus_em = {}
    penalty_time = [0.1, 0.5, 1, 5]
    penalty_mark = [0.1, 0.5, 1, 5]
    omega = [0.2, 0.5, 1]
    v = [0.2, 0.5, 1]

    for _penalty_time, _penalty_mark, _omega, _v in itertools.product(penalty_time, penalty_mark, omega, v):
        for filename in filenames:
            print("penalty_time:", _penalty_time, ", penalty_mark:", _penalty_mark, ", omega:", _omega, ", v:", _v)
            print(filename)
            print("-----------------------------------------------------------------")

            # penalty_time, penalty_mark = 0.1, 0.1
            with open('./assets/data/%s.pickle' % filename, 'rb') as handle:
                timestamps, timestamp_dims, edge, mark = pickle.load(handle)
            fake_num = int(0.4 * len(timestamps))
            T = max(timestamps)
            n_dim = len(np.unique(timestamp_dims))
            sentiments = np.sort(np.unique(mark))
            exo_T = T
            fake = exo_by_num(0, exo_T, n_dim, fake_num)
            n_mark = len(np.unique(mark))
            fake_dim = [f[0] for f in fake]
            fake_time = [f[1] for f in fake]
            fake_mark = np.random.choice(sentiments, len(fake_time))
            timestamp_dims = np.concatenate((timestamp_dims, fake_dim))
            timestamps = np.concatenate((timestamps, fake_time))
            mark = np.concatenate((mark, fake_mark))
            sort_idx = np.argsort(timestamps)
            timestamp_dims, timestamps, mark = timestamp_dims[sort_idx], timestamps[sort_idx], mark[sort_idx]

            print("filename: %s" % filename)
            print("fake_num: %s" % fake_num)
            real_exo_idxs = np.where(np.isin(timestamps, fake_time) == True)[0]
            exo_mask = np.full_like(timestamps, False)
            exo_mask[real_exo_idxs] = True
            real_exo_idxs = sort_idx[len(timestamps) - fake_num:]

            # Our method
            pred_exo_idxs_ours, incr_fns = hybrid_greedy(timestamps, timestamp_dims, mark, iter, _omega, _v, n_dim, T,
                                                         _penalty_time, \
                                                         _penalty_mark, edge, sentiments,
                                                         stochastic_size=stochastic_size, verbose=True, return_all=True)

            ours_exo_mask = np.full_like(timestamps, False)
            ours_exo_mask[pred_exo_idxs_ours] = True
            accu_ours = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_ours])
            accus_ours[filename] = accu_ours
            print("ours: %s" % accus_ours[filename])

            # facloc
            pred_exo_idxs_facloc = facloc_pred(timestamps, timestamp_dims, mark, iter)
            accu_facloc = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_facloc])
            accus_facloc[filename] = accu_facloc
            print("facloc: %s" % accus_facloc[filename])

            # kmeans
            pred_exo_idx_kmeans = kmeans_pred(timestamps, timestamp_dims, mark, iter)
            accu_kmeans = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_kmeans])
            accus_kmeans[filename] = accu_kmeans
            print("kmeans: %s" % accus_kmeans[filename])

            # PCA
            pred_exo_idx_pca = pca_pred(timestamps, timestamp_dims, mark, iter, k=1)
            accu_pca = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_pca])
            accus_pca[filename] = accu_pca
            print("pca: %s" % accus_pca[filename])

            # EM
            prex_exo_idx_em = em_pred(timestamps, timestamp_dims, mark, iter, edge, _omega, _v, n_dim, sentiments, T)
            accu_em = np.mean([(1 if _ in real_exo_idxs else 0) for _ in prex_exo_idx_em])
            accus_em[filename] = accu_em
            print("em: %s" % accus_em[filename])

            print("-----------------------------------------------------------------")

        print("accus_ours")
        print(accus_ours)
        print("accus_facloc")
        print(accus_facloc)
        print("accus_kmeans")
        print(accus_kmeans)
        print("accus_pca")
        print(accus_pca)
        print("accus_em")
        print(accus_em)
