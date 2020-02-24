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

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(243, 10, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)


#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {}
fp_probas = {}

#Load up the dictionaries with the relevant feature matrices for each fingerprint:
for fp in fp_names:
#    ####
#    ###Doing a direct comparison with CATS here:
#    ####
#    if fp not in ['morgan', 'cats']:
#        continue
    print(fp)
    print('Loading:', fp)
    featureMatrix, labels = utils.load_feature_and_label_matrices(type=fp)
    featureMatrix_, labels__ = utils.get_subset(featureMatrix, y, indices=col_indices)
    fp_dict[fp]=sparse.csr_matrix(featureMatrix_)
    fp_probas[fp] = []


#This will be used for clustering:
distance_matrix = utils.fast_dice(x_)


print('')
print('The below:')
print('- chooses a distance threshold in a reasonable range (from the distribution ~ U(low=0.05, high=0.425))')
print('- clusters the dataset using single-linkage agglomerative clustering')
print('- generates a train/test split based on random assignment of clusters and a randomly selected target,')
print('- calculates AVE bias for that split,')
print('- and, for each fingerprint, makes probability predictions for the test set')
print('\tbased on a logistic regression classifier.')
print('')


#These will be used to save all the data so we don't have to repeatedly run this script
test_labels = list()
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(450)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random clustering cutoff and cluster using the original morgan FPs:
    cutoff = np.random.uniform(0.3, 0.7)
    clust = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage='single', affinity='precomputed')
    clust.fit(distance_matrix)
    
    #Get the train test split:
    #this time, we first split the labels outside the utils function
    #so that all fingerprints can recieve exactly the same split:
    percentage_holdout = 0.2
    test_clusters = np.random.choice(clust.labels_.max(), int(clust.labels_.max()*percentage_holdout), replace=False)
    x_train, x_test, y_train, y_test = utils.make_cluster_split(x_, y_, clust, test_clusters=test_clusters)

    #ensure there is enough data for each class in each split:
    num_train_pos = (y_train[:,idx]==1).sum()
    num_test_pos = (y_test[:,idx]==1).sum()
    num_train_neg = (y_train[:,idx]==0).sum() 
    num_test_neg = (y_test[:,idx]==1).sum()
    if min(num_train_pos, num_test_pos, num_train_neg, num_test_neg)>30:
        
    	#the feature matrices:
        matrices = utils.split_feature_matrices(x_train, x_test, y_train, y_test, idx)
    	#pairwise distance matrices between all the above in 'martices'
        distances = utils.calc_distance_matrices(matrices)
    	#calc the AVE bias:
        AVE = utils.calc_AVE(distances)

        for fp in fp_names:
 #           if fp not in ['morgan', 'cats']:
 #               continue

            x_train, x_test, y_train, y_test = utils.make_cluster_split(fp_dict[fp], y_, clust, test_clusters=test_clusters)
            #Fit some ML model (can be anything - logreg here):
            clf = LogisticRegression(solver='lbfgs', max_iter=1000)
            clf.fit(sparse.csr_matrix(x_train), y_train[:,idx])
            #make probaility predictions for the positive class:
            proba = clf.predict_proba(x_test)[:,1]
            fp_probas[fp].append(proba)

	##Add all the data to our lists:
        aves.append(AVE)
        sizes.append([i.shape[0] for i in matrices])
        targets.append(idx)
        cutoffs.append(cutoff)
        test_labels.append(y_test[:,idx])


##Save all the AVEs and model prediction data:
np.save('./processed_data/fp_comparison/aves.npy', np.array(aves))
np.save('./processed_data/fp_comparison/sizes.npy', np.array(sizes))
np.save('./processed_data/fp_comparison/targets.npy', np.array(targets))
np.save('./processed_data/fp_comparison/cutoffs.npy', np.array(cutoffs))
np.save('./processed_data/fp_comparison/test_labels.npy', np.array(test_labels))

for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue
    np.save('./processed_data/fp_comparison/'+fp+'_probas.npy', np.array(fp_probas[fp]))
