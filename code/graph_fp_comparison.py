import utils
from os import path

import numpy as np
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform

from paris_cluster import ParisClusterer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(243, 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)


print('')
print('The below:')
print('- generates a weighted, directed nearest neighbor graph using PyNNDescent')
print('- generates a linkage tree using the Paris clustering algorithm')
print('- clusters by cutting the tree in balanced cuts of size in range giving low AVE')
print('- generates a train/test split based on random assignment of clusters and a randomly selected target,')
print('- calculates AVE bias for that split,')
print('- and generates predicted probabilities on the test set for multiple fingerprints using logistic regression')
print('')




#build the hierarchical clustering tree:
clusterer = ParisClusterer(x_.toarray())
clusterer.buildAdjacency()
clusterer.fit()


#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {}
fp_probas = {}

#Load up the dictionaries with the relevant feature matrices for each fingerprint:
for fp in fp_names:
    print(fp)
    featureMatrix, labels = utils.load_feature_and_label_matrices(type=fp)
    featureMatrix_, labels__ = utils.get_subset(featureMatrix, y, indices=col_indices)
    fp_dict[fp]=sparse.csr_matrix(featureMatrix_)
    fp_probas[fp] = []


#These will be used to save all the data so we don't have to repeatedly run this script
test_labels = list()
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(150)):
    clusterSize = np.random.randint(3000,5000)
    clusterer.balanced_cut(clusterSize)

    for repeat in range(1):
        #choose a random target:
        idx = np.random.choice(y_.shape[1])

        #Get the train test split:
        pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
        neg_labels = np.unique(clusterer.labels_[y_[:,idx]!=1])
        test_clusters = list(np.random.choice(pos_labels, int(0.2*len(pos_labels)), replace=False))+list(np.random.choice(neg_labels, int(0.2*len(neg_labels)), replace=False))

        x_train, x_test, y_train, y_test = utils.make_cluster_split(x_, y_, clusterer, test_clusters=test_clusters)

        #ensure there is enough data for each class in each split:
        num_train_pos = (y_train[:,idx]==1).sum()
        num_test_pos = (y_test[:,idx]==1).sum()
        num_train_neg = (y_train[:,idx]==0).sum() 
        num_test_neg = (y_test[:,idx]==1).sum()
        if min(num_train_pos, num_test_pos, num_train_neg, num_test_neg)>50:

    	    #the feature matrices:
#            matrices = utils.split_feature_matrices(x_train, x_test, y_train, y_test, idx)
    	    #pairwise distance matrices between all the above in 'martices'
#            distances = utils.calc_distance_matrices(matrices)
    	    #calc the AVE bias:
#            AVE = utils.calc_AVE(distances)

#            for fp in fp_names:
#                print(f'Fitting {fp}', end='-')
#                x_train, x_test, y_train, y_test = utils.make_cluster_split(fp_dict[fp], y_, clusterer, test_clusters=test_clusters)
#                #Fit some ML model (can be anything - logreg here):
#                clf = LogisticRegression(solver='lbfgs', max_iter=1000)
#                clf.fit(sparse.csr_matrix(x_train), y_train[:,idx])
#                #make probaility predictions for the positive class:
#                proba = clf.predict_proba(x_test)[:,1]
#                fp_probas[fp].append(proba)

            ##Add all the data to our lists:
            #aves.append(AVE)
#            sizes.append([i.shape[0] for i in matrices])
            targets.append(idx)
            cutoffs.append(clusterSize)
            test_labels.append(y_test[:,idx])


##Save all the AVEs and model prediction data:
#np.save('./processed_data/graph_fp_comparison/aves.npy', np.array(aves))
#np.save('./processed_data/graph_fp_comparison/sizes.npy', np.array(sizes))
#np.save('./processed_data/graph_fp_comparison/targets.npy', np.array(targets))
#np.save('./processed_data/graph_fp_comparison/cutoffs.npy', np.array(cutoffs))
np.save('./processed_data/graph_fp_comparison/test_labels.npy', np.array(test_labels))

#for fp in fp_names:
#    np.save('./processed_data/graph_fp_comparison/'+fp+'_probas.npy', np.array(fp_probas[fp]))
