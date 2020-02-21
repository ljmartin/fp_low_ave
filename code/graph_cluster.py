import utils
from os import path

import numpy as np
from scipy import stats, sparse

from paris_cluster import ParisClusterer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(243, 10, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)



print('')
print('The below:')
print('- generates a weighted, directed nearest neighbor graph using PyNNDescent')
print('- generates a linkage tree using the Paris clustering algorithm')
print('- clusters by cutting the tree in balanced cuts of size U(low=200, high=2000)')
print('- generates a train/test split based on random assignment of clusters and a randomly selected target,')
print('- calculates AVE bias for that split,')
print('')


clusterer = ParisClusterer(x_)
clusterer.buildAdjacency()
clusterer.fit()


#These will be used to save all the data so we don't have to repeatedly run this script
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(5)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random clustering cutoff and cluster:
    clusterSize = np.random.randint(200,2500)
    clusterer.balancedCut(clusterSize)
    
    #Get the train test split:
    x_train, x_test, y_train, y_test = utils.make_cluster_split(x_, y_, clusterer)

    #ensure you have enough positive ligands for this target in both the train / test folds: 
    num_train_pos = (y_train[:,idx]==1).sum()
    num_test_pos = (y_test[:,idx]==1).sum()
    num_train_neg = (y_train[:,idx]==0).sum() 
    num_test_neg = (y_test[:,idx]==1).sum()
    #ensure there is enough data for each class in each split:
    if min(num_train_pos, num_test_pos, num_train_neg, num_test_neg)>10:

    	#the feature matrices:
        matrices = utils.split_feature_matrices(x_train, x_test, y_train, y_test, idx)
    	#pairwise distance matrices between all the above in 'martices'
        distances = utils.calc_distance_matrices(matrices)
    	#calc the AVE bias:
        AVE = utils.calc_AVE(distances)

	##Add all the data to our lists:
        aves.append(AVE)
        sizes.append([i.shape[0] for i in matrices])
        targets.append(idx)
        cutoffs.append(cutoff)

##Save all the AVEs and model prediction data:
np.save('./processed_data/graph_cluster/aves.npy', np.array(aves))
np.save('./processed_data//sizes.npy', np.array(sizes))
np.save('./processed_data/graph_cluster/targets.npy', np.array(targets))
np.save('./processed_data/graph_cluster/cutoffs.npy', np.array(cutoffs))


