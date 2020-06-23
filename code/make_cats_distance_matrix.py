import utils
from os import path

import numpy as np
from scipy import stats, sparse
from tqdm import tqdm

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(226, 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)



from scipy.spatial.distance import cdist
import numba

cats_distance_matrix = np.memmap('./processed_data/distance_matrices/cats_distance_matrix.dat', 
                                 dtype='float16', 
                                 mode='w+', 
                                 shape=(x_.shape[0],x_.shape[0]))


catsMatrix, _ = utils.load_feature_and_label_matrices(type='cats')
catsMatrix_, __ = utils.get_subset(catsMatrix, y, indices=col_indices)
catsMatrix_ = catsMatrix_.toarray()

#https://stackoverflow.com/questions/47315659/using-numba-for-cosine-similarity-between-a-vector-and-rows-in-a-matix
@numba.guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
def fast_cosine_gufunc(u, v, result):
  
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    result[:] = ratio

for item in tqdm(range(catsMatrix_.shape[0])):
    distances = np.array(np.clip((1-fast_cosine_gufunc(catsMatrix_[item], catsMatrix_[:]))/2,0,1))**0.25
    #distances = ((1-fast_cosine_gufunc(catsMatrix_[item], catsMatrix_[:]))/2)**0.25
    cats_distance_matrix[item]=distances

#remove the writeable object:
del cats_distance_matrix
