import utils
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

#load the pairwise distance matrix:
cats_distance_matrix = np.memmap('./processed_data/distance_matrices/cats_distance_matrix.dat', dtype=np.float16, mode='r', 
                                 shape=(x_.shape[0], x_.shape[0]))

#this sparse matrix is the adjacency graph
#using DOK matrix because it's faster to write. It is converted to CSR after. 
wdAdj = sparse.dok_matrix((x_.shape[0], x_.shape[0]), dtype=float)

#iterate through every row, writing adjacencies for the 50-NN.
for row in tqdm(range(cats_distance_matrix.shape[0])):
    all_distances = cats_distance_matrix[row]

    neighbors = np.argpartition(all_distances, 50)[:50]
    distances = all_distances[neighbors]
    
    for neighbourIndex, distance in zip(neighbors[1:], distances[1:]):
        wdAdj[row, neighbourIndex] += 1-distance # because similarity is 1-distance, and this a weighted adjacency



sparse.save_npz('./processed_data/distance_matrices/wadj_cats.npz', sparse.csr_matrix(wdAdj))
