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

#Open a memory mapped distance matrix.
#We do this because the pairwise distance matrix for 100 targets does not fit in memory.
#It is nearly 100% dense and has 117747*117747 = 13864356009 elements. This is also
#Why it uses float16 (reducing the required storage space to ~26GB, c.f. 52GB for float32).
distance_matrix = np.memmap('./processed_data/graph_fp_comparison/distMat.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))



#build the hierarchical clustering tree:
clusterer = ParisClusterer(x_.toarray())
clusterer.buildAdjacency()
clusterer.fit()

#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {} #this stores the actual fingerprint feature matrices
fp_ap = {} #this stores the average precisions

#Load up the dictionaries with the relevant feature matrices for each fingerprint:
for fp in fp_names:
    print(fp)
    featureMatrix, labels = utils.load_feature_and_label_matrices(type=fp)
    featureMatrix_, labels__ = utils.get_subset(featureMatrix, y, indices=col_indices)
    fp_dict[fp]=sparse.csr_matrix(featureMatrix_)
    fp_ap[fp] = []


#Store all the results in these:
targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(2)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random cluster size upper limit and cluster:
    clusterSize = np.random.randint(200,10000)
    clusterer.balanced_cut(clusterSize)

    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
    neg_labels = clabels[~np.isin(clabels, pos_labels)]
    if min(len(pos_labels), len(neg_labels))<2:
        print('Not enough positive clusters to split')
        continue

    test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.2, [0.1,0.1], shuffle=True)

    actives_test_idx, actives_train_idx, inactives_test_idx, inactives_train_idx = utils.get_four_matrices(y_,idx,clusterer,test_clusters,train_clusters)
    print(actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0])
    print(min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]]))        
    if min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]])<20:        
           print('Not enough ligands to train and test')
           continue
    #No need to calculate the AVE beforehand this time.
    #ave= utils.calc_AVE_quick(distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    #aves_before_trim.append(ave)
    
    #Now we will trim some nearest neighbours and by doing so, reduce AVE.
    #trim from the inactives/train matrix first:
    inactive_dmat = distance_matrix[inactives_test_idx]
    print('New inactives train_idx', inactive_dmat.shape, inactives_train_idx.shape, inactives_test_idx.shape)
    new_inactives_train_idx = utils.trim(inactive_dmat, 
                                       inactives_train_idx, 
                                       inactives_test_idx,
                                             fraction_to_trim=0.2)
    #then trim from the actives/train matrix:
    active_dmat = distance_matrix[actives_test_idx]
    print('New actives train_idx', active_dmat.shape, actives_train_idx.shape, actives_test_idx.shape)
    new_actives_train_idx = utils.trim(active_dmat,
                                    actives_train_idx, 
                                    actives_test_idx,
                                     fraction_to_trim=0.2)

    #now calculate AVE with this new split:
    ave= utils.calc_AVE_quick(distance_matrix, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx)
    aves_after_trim.append(ave)

    print('Fitting...')
    for fp in fp_names:
        results = utils.evaluate_split(fp_dict[fp], y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True)
        fp_ap[fp].append(results['ap'])
    print('Done')

    cutoffs.append(clusterSize)
    targets.append(idx)
    sizes.append([new_actives_train_idx.shape[0], actives_test_idx.shape[0], new_inactives_train_idx.shape[0], inactives_test_idx.shape[0]])

##Save all the AVEs and model prediction data:
np.save('./processed_data/graph_fp_comparison/aves.npy', np.array(aves))
np.save('./processed_data/graph_fp_comparison/sizes.npy', np.array(sizes))
np.save('./processed_data/graph_fp_comparison/targets.npy', np.array(targets))
np.save('./processed_data/graph_fp_comparison/cutoffs.npy', np.array(cutoffs))
