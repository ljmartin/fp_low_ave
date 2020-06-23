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



morgan_distance_matrix = np.memmap('./processed_data/distance_matrices/morgan_distance_matrix.dat', 
                                 dtype='float16', 
                                 mode='w+', 
                                 shape=(x_.shape[0],x_.shape[0]))

top =int(np.ceil(x_.shape[0]/1000))
print(top, x_.shape)
for item in tqdm(range(top)):
    start = item*1000
    end = min(x_.shape[0], start+1000)
    distances = utils.fast_dice(x_[start:end], x_[:])
    morgan_distance_matrix[start:end]=distances

#remove the writeable object:
del morgan_distance_matrix
