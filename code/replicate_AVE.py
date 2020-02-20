import utils
from os import path

import numpy as np
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())


##The J&J benchmark used single-linkage clustering. 
#Wallach et. al later showed how model performance was related to bias. 
#Here we reproduce this using a set of train/test splits
#also using single-linkage clustering.

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(243, 10, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)


#This will be used for clustering:
distance_matrix = utils.fast_dice(x_)


print('')
print('The below:')
print('- chooses a distance threshold in a reasonable range (from the distribution ~ U(low=0.05, high=0.425))')
print('- clusters the dataset using single-linkage agglomerative clustering')
print('- generates a train/test split based on random assignment of clusters and a randomly selected target,')
print('- calculates AVE bias for that split,')
print('- and makes probability predictions for the test set based on a logistic regression classifier.')
print('')


#These will be used to save all the data so we don't have to repeatedly run this script
test_labels = list()
test_probas = list()
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(300)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random clustering cutoff and cluster:
    cutoff = stats.uniform(0.05, 0.7).rvs()
    clust = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage='single', affinity='precomputed')
    clust.fit(distance_matrix)
    
    #Get the train test split:
    x_train, x_test, y_train, y_test = utils.make_cluster_split(x_, y_, clust)

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
        #calc the VE bias;
        #VE = utils.calc_VE(distances)

        #Fit some ML model (can be anything - logreg here):
        clf = LogisticRegression(solver='lbfgs', max_iter=500)
        clf.fit(sparse.csr_matrix(x_train), y_train[:,idx])
        #make probaility predictions for the positive class:
        proba = clf.predict_proba(x_test)[:,1]

	##Add all the data to our lists:
        aves.append(AVE)
        sizes.append([i.shape[0] for i in matrices])
        #ves.append(VE)
        targets.append(idx)
        cutoffs.append(cutoff)
        test_probas.append(proba)
        test_labels.append(y_test[:,idx])


##Save all the AVEs and model prediction data:
np.save('./processed_data/replicate_AVE/aves.npy', np.array(aves))
np.save('./processed_data/replicate_AVE/sizes.npy', np.array(sizes))
#np.save('./processed_data/replicate_AVE/ves.npy', np.array(ves))
np.save('./processed_data/replicate_AVE/targets.npy', np.array(targets))
np.save('./processed_data/replicate_AVE/cutoffs.npy', np.array(cutoffs))
np.save('./processed_data/replicate_AVE/test_probas.npy', np.array(test_probas))
np.save('./processed_data/replicate_AVE/test_labels.npy', np.array(test_labels))

