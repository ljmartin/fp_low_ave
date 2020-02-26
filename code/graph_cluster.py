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
col_indices = np.random.choice(243, 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)

#Open a memory mapped distance matrix.
#We do this because the pairwise distance matrix for 100 targets does not fix in memory!
#It is nearly 100% dense and has 117747*117747 = 13864356009 elements. This is also
#Why it uses float16 (reducing storage space required to ~26GB).
dist_mapped = np.memmap('./processed_data/graph_fp_comparison/distMat.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))

print('')
print('The below:')
print('- generates a weighted, directed nearest neighbor graph using PyNNDescent')
print('- generates a linkage tree using the Paris clustering algorithm')
print('- clusters by cutting the tree in balanced cuts of size U(low=200, high=5000)')
print('- generates a train/test split based on random assignment of clusters and a randomly selected target,')
print('- calculates AVE bias for that split,')
print('')


clusterer = ParisClusterer(x_.toarray())
clusterer.buildAdjacency()
clusterer.fit()


#These will be used to save all the data so we don't have to repeatedly run this script
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(30)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random clustering cutoff and cluster:
    clusterSize = np.random.randint(200,10000)
    clusterer.balanced_cut(clusterSize)

    #Get all cluster labels, extract any cluster associated with positives
    #(the rest must be all-negative), then shuffle these labels
    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
    num_pos_clusters = len(pos_labels)
    neg_labels = clabels[~np.isin(clabels, pos_labels)]
    num_neg_clusters = len(neg_labels)
    np.random.shuffle(pos_labels)
    np.random.shuffle(neg_labels)
    ##Make a split based on 50/50 of the positive clusters and 10/90 of the
    ##negative clusters:
    #splitting positives in half:
    test_pos_clusters = pos_labels[:int(num_pos_clusters/2)]
    train_pos_clusters = pos_labels[int(num_pos_clusters/2):]
    #taking first 10% and last 90% of negatives:
    test_neg_clusters = neg_labels[:int(num_neg_clusters/10)]
    train_neg_clusters = neg_labels[int(num_neg_clusters/10):]
    #combined:
    test_clusters = list(test_pos_clusters)+list(test_neg_clusters)
    train_clusters = list(train_pos_clusters)+list(train_neg_clusters)

    #Now we can get indices for the test and train instances:
    alltest, alltrain, allpos, allneg = utils.get_split_indices(y_, idx, clusterer, test_clusters, train_clusters)
    actives_test_indices = (alltest&allpos).nonzero()[0]
    actives_train_indices = (alltrain&allpos).nonzero()[0]
    inactives_test_indices = (alltest&allneg).nonzero()[0]
    inactives_train_indices = (alltrain&allneg).nonzero()[0]

    #These are the distances between inactive test ligands and all other ligands:
    inactive_test_train_dmat = dist_mapped[inactives_test_indices]
    #we trim the first 10,000 ligands in the inactive/training set, ordered by closeness to the inactive/test set.
    new_inactives_train_indices = inactives_train_indices[inactive_test_train_dmat[:,inactives_train_indices].min(0).argsort()[10000:]]

    #Now we can calculate AVE:
    iTest_iTrain_D = inactive_test_train_dmat[:,new_inactives_train_indices].min(1)
    iTest_aTrain_D = inactive_test_train_dmat[:,actives_train_indices].min(1)
    actives = dist_mapped[actives_test_indices]
    aTest_aTrain_D = actives[:,actives_train_indices].min(1)
    aTest_iTrain_D = actives[:,new_inactives_train_indices].min(1)

    aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
    aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
    iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
    iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )

    ave = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
    
    aves.append(ave)
    sizes.append([len(actives_train_indices), len(actives_test_indices), len(new_inactives_train_indices), len(inactives_test_indices)])
    targets.append(idx)
    cutoffs.append(clusterSize)

##Save all the AVEs and model prediction data:
np.save('./processed_data/graph_cluster/aves.npy', np.array(aves))
np.save('./processed_data/graph_cluster/sizes.npy', np.array(sizes))
np.save('./processed_data/graph_cluster/targets.npy', np.array(targets))
np.save('./processed_data/graph_cluster/cutoffs.npy', np.array(cutoffs))


