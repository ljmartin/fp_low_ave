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
col_indices = np.random.choice(226, 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)

#Open a memory mapped distance matrix.
#We do this because the pairwise distance matrix for 100 targets does not fit in memory.
#It is nearly 100% dense and has 117747*117747 = 13864356009 elements. This is also
#Why it uses float16 (reducing the required storage space to ~26GB, c.f. 52GB for float32).
distance_matrix = np.memmap('./processed_data/distance_matrices/morgan_distance_matrix.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))

distance_matrix = np.memmap('./processed_data/distance_matrices/cats_distance_matrix.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))


clusterer = ParisClusterer(x_.toarray())
clusterer.loadAdjacency('./processed_data/distance_matrices/wadj_ecfp.npz')
clusterer.fit()


#Store all the results in these:
aves_before_trim = list()
aves_after_trim = list()
ap_before_trim = list()
ap_after_trim = list()
mcc_before_trim = list()
mcc_after_trim = list()
sizes_before_trim = list()
sizes_after_trim = list()


targets = list()
cutoffs = list()
aves = list()
sizes = list()

for _ in tqdm(range(400)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random cluster size upper limit and cluster:
    clusterSize = np.random.randint(100,7500)
    clusterer.labels_ = utils.cut_balanced(clusterer.paris.dendrogram_, clusterSize)

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
    ave= utils.calc_AVE_quick(distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    aves_before_trim.append(ave)
    sizes_before_trim.append([actives_train_idx.shape[0], actives_test_idx.shape[0], inactives_train_idx.shape[0], inactives_test_idx.shape[0]])
    
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

    #evaluate a LogReg model using the original single-linkage split
    results = utils.evaluate_split(x_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=True)
    ap_before_trim.append(results['ap'])
    mcc_before_trim.append(results['mcc'])

    #evaluate a LogReg model using the new (lower AVE) split:
    results = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=True)
    ap_after_trim.append(results['ap'])
    mcc_after_trim.append(results['mcc'])
    
    cutoffs.append(clusterSize)
    sizes_after_trim.append([new_actives_train_idx.shape[0], actives_test_idx.shape[0], new_inactives_train_idx.shape[0], inactives_test_idx.shape[0]])
    targets.append(idx)

    ##Save all the AVEs and model prediction data:
    np.save('./processed_data/graph_cluster_both/aves_before_trim.npy', np.array(aves_before_trim))
    np.save('./processed_data/graph_cluster_both/aves_after_trim.npy', np.array(aves_after_trim))
    np.save('./processed_data/graph_cluster_both/ap_before_trim.npy', np.array(ap_before_trim))
    np.save('./processed_data/graph_cluster_both/ap_after_trim.npy', np.array(ap_after_trim))
    np.save('./processed_data/graph_cluster_both/mcc_before_trim.npy', np.array(mcc_before_trim))
    np.save('./processed_data/graph_cluster_both/mcc_after_trim.npy', np.array(mcc_after_trim))
    np.save('./processed_data/graph_cluster_both/sizes.npy', np.array(sizes))
    np.save('./processed_data/graph_cluster_both/targets.npy', np.array(targets))
    np.save('./processed_data/graph_cluster_both/cutoffs.npy', np.array(cutoffs))

