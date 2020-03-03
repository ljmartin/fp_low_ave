import utils
from os import path

import numpy as np
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform, cdist

import timeit, functools
import matplotlib.pyplot as plt

##Set plotting parameters:
utils.set_mpl_params()

##Load some instance data (it starts off being sparse)
x, y = utils.load_feature_and_label_matrices(type='morgan')


#calculates distances using scipy's cdist
def dense_cdist(arr1, arr2, metric):
    cdist(arr1, arr2, metric=metric)
    pass

#calculates distances using sparse matrices
def sparse_dist(arr1, arr2, func):
    func(arr1, arr2)
    pass


number_calcs = np.array([10,30, 100, 300, 1000, 3000])

dense_times_jaccard = list()
sparse_times_jaccard = list()
dense_times_dice = list()
sparse_times_dice = list()

for arr_size in number_calcs:
    arr = x[:arr_size].toarray()
    ##calculate timings for dense arrays using cdist:
    t = timeit.Timer(functools.partial(dense_cdist, arr, arr, 'jaccard')) 
    dense_times_jaccard.append(t.timeit(5)/5)
    t = timeit.Timer(functools.partial(dense_cdist, arr, arr, 'dice')) 
    dense_times_dice.append(t.timeit(5)/5)

    ##calculate timings for sparse arrays:
    t = timeit.Timer(functools.partial(sparse_dist, x[:arr_size], x[:arr_size], utils.fast_jaccard)) 
    sparse_times_jaccard.append(t.timeit(5)/5)
    t = timeit.Timer(functools.partial(sparse_dist, x[:arr_size], x[:arr_size], utils.fast_dice)) 
    sparse_times_dice.append(t.timeit(5)/5)


fig, ax = plt.subplots(1,2)
fig.set_figwidth(10)
fig.set_figheight(5)
ax[0].plot(number_calcs, dense_times_jaccard, '-o', label='Dense, Jaccard')
ax[0].plot(number_calcs, sparse_times_jaccard, '-o', label='Sparse, Jaccard')
ax[0].set_xlabel('Number of instances compared')
ax[0].set_ylabel('Time taken')
utils.plot_fig_label(ax[0],'A.')
ax[0].legend()
ax[0].grid()


ax[1].plot(number_calcs, dense_times_dice, '-o', label='Dense, Dice')
ax[1].plot(number_calcs, sparse_times_dice, '-o', label='Sparse, Dice')

ax[1].set_xlabel('Number of instances compared')
ax[1].set_ylabel('Time taken')
utils.plot_fig_label(ax[1],'B.')
ax[1].legend()
ax[1].grid()

fig.savefig('./processed_data/supplementary/time_comparison.png')
plt.close(fig)
