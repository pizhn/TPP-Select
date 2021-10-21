from itertools import product

from baseline.pca import pca_pred
from baseline.em import em_pred
from baseline.facloc import facloc_pred
from baseline.kmeans import kmeans_pred
from func.generate_synthetic import gen_synthetic_with_mark, process_history
import numpy as np
# from autodump import Autodumper
import time

from greedy.greedy import hybrid_greedy

if __name__ == '__main__':
    dim, num_message = [5], [310]
    omega, v = [0.1, 0.5, 1], [0.1, 0.5, 1]
    num_sentiments = 2
    # frac_exo = [0.2]
    frac_exo = [0.1, 0.3, 0.5, 0.7]
    T = 100
    penalty_time, penalty_mark = [0.1, 0.5, 1], [0.1, 0.5, 1]

    hyperparams = np.random.permutation(list(product(dim, num_message, omega, v, frac_exo, penalty_time, penalty_mark)))

    accu_ours, accu_facloc, accu_kmeans, accu_pca, accu_em = [], [], [], [], []

    for _dim, _num_message, _omega, _v, _frac_exo, _penalty_time, _penalty_mark in hyperparams:
        _dim, _num_message = int(_dim), int(_num_message)
        exo_size = int(_num_message * _frac_exo)
        sentiments = np.linspace(-1, 1, num_sentiments)
        print("dim:", _dim)
        print("num_message:", _num_message)
        print("omega:", _omega)
        print("v:", _v)
        print("frac_exo:", _frac_exo)
        print("penalty_time:", _penalty_time)
        print("penalty_mark:", _penalty_mark)
        print("dim:", _dim)

        edge, history, mark, beta_real = gen_synthetic_with_mark(_dim, _num_message, _frac_exo, _omega, _v, T,
                                                                 num_sentiments, start_from_zero=False)
        timestamps, timestamp_dims, true_endo_mask = process_history(history)
        real_exo_idxs = np.where(true_endo_mask == False)[0]

        start = time.time()

        real_endo = np.full(len(timestamps), True)
        real_endo[real_exo_idxs] = False

        # ours
        pred_exo_idxs_ours = hybrid_greedy(timestamps, timestamp_dims, mark, exo_size, _omega, _v, _dim, T,
                                           _penalty_time, _penalty_mark, edge, sentiments,
                                           stochastic_size=None, verbose=True)
        pred_endo_ours = np.full(len(timestamps), True)
        pred_endo_ours[pred_exo_idxs_ours] = False
        accu_ours += [np.mean(real_endo == pred_endo_ours)]
        print("ours:", accu_ours[-1])

        # facloc
        pred_exo_idxs_facloc = facloc_pred(timestamps, timestamp_dims, mark, exo_size)
        pred_endo_facloc = np.full(len(timestamps), True)
        pred_endo_facloc[pred_exo_idxs_facloc] = False
        accu_facloc += [np.mean(real_endo == pred_endo_facloc)]
        print("facloc:", accu_facloc[-1])

        # kmeans
        pred_exo_idx_kmeans = kmeans_pred(timestamps, timestamp_dims, mark, exo_size)
        pred_endo_kmeans = np.full(len(timestamps), True)
        pred_endo_kmeans[pred_exo_idx_kmeans] = False
        accu_kmeans += [np.mean(real_endo == pred_endo_kmeans)]
        print("kmeans:", accu_kmeans[-1])

        # PCA
        pred_exo_idx_pca = pca_pred(timestamps, timestamp_dims, mark, exo_size)
        pred_endo_pca = np.full(len(timestamps), True)
        pred_endo_pca[pred_exo_idx_kmeans] = False
        accu_pca += [np.mean(real_endo == pred_endo_pca)]
        print("pca:", accu_pca[-1])

        # EM
        pred_exo_idx_em = em_pred(timestamps, timestamp_dims, mark, exo_size, edge, _omega, _v, _dim, sentiments, T)
        pred_endo_em = np.full(len(timestamps), True)
        pred_endo_em[pred_exo_idx_kmeans] = False
        accu_em += [np.mean(real_endo == pred_endo_em)]
        print("em:", accu_em[-1])
