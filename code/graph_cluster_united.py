import utils
from os import path
import numpy as np
from scipy import stats, sparse
from paris_cluster import ParisClusterer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import pandas as pd

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(226, 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)

#load CATS features as well:
catsMatrix, _ = utils.load_feature_and_label_matrices(type='cats')
catsMatrix_, __ = utils.get_subset(catsMatrix, y, indices=col_indices)


#Open a memory mapped distance matrix.
#We do this because the pairwise distance matrix for 100 targets does not fit in memory.
#It is nearly 100% dense and has 117747*117747 = 13864356009 elements. This is also
#Why it uses float16 (reducing the required storage space to ~26GB, c.f. 52GB for float32).
morgan_distance_matrix = np.memmap('./processed_data/distance_matrices/morgan_distance_matrix.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))
cats_distance_matrix = np.memmap('./processed_data/distance_matrices/cats_distance_matrix.dat', dtype=np.float16,
              shape=(x_.shape[0], x_.shape[0]))


#clusterer = ParisClusterer(x_.toarray())
#clusterer.loadAdjacency('./processed_data/distance_matrices/wadj_ecfp.npz')
#clusterer.fit()


#Store all the results in these:
df_after_trim = pd.DataFrame(columns = ['ave_cats', 'ave_morgan', 'ap_cats', 'ap_morgan'])


targets = list()
cutoffs = list()
aves = list()
sizes = list()

loc_counter = 0

for _ in tqdm(range(1500), smoothing=0):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])
    positives = (y_[:,idx]==1).nonzero()[0]

    #choose a random cluster size upper limit and cluster:
    clusterer = ParisClusterer(x_)
    clusterer.wdAdj = sparse.csr_matrix(morgan_distance_matrix[positives][:,positives])
    clusterer.fit()
    clusterSize = np.random.randint(100,400)
    clusterer.labels_ = utils.cut_balanced(clusterer.paris.dendrogram_, clusterSize)

    #cutoff = 0.41
    #clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage='single', affinity='precomputed')    

    #clusterer.fit(morgan_distance_matrix[positives][:,positives])
    print(np.unique(clusterer.labels_, return_counts=True))

    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_)
    #neg_labels = clabels[~np.isin(clabels, pos_labels)]
    np.random.shuffle(pos_labels)
    num_pos_clusters = len(pos_labels)
    pos_test_fraction = 0.1
    print(num_pos_clusters)
    test_pos_clusters = pos_labels[:max(1,round(num_pos_clusters*pos_test_fraction))]
    train_pos_clusters = pos_labels[max(1,round(num_pos_clusters*pos_test_fraction)):]
    print(len(test_pos_clusters), len(train_pos_clusters))
    actives_test_idx = np.isin(clusterer.labels_, test_pos_clusters).nonzero()[0]
    actives_train_idx = np.isin(clusterer.labels_, train_pos_clusters).nonzero()[0]
    print(len(actives_test_idx), len(actives_train_idx))
    



    
    if len(pos_labels)<2:
    #if min(len(pos_labels), len(neg_labels))<2:
        print('Not enough positive clusters to split')
        continue

    #####
    ##Step1: cluster
    #####
    #test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.1, [0.1,0.1], shuffle=True)
    #actives_test_idx, actives_train_idx, inactives_test_idx, inactives_train_idx = utils.get_four_matrices(y_,idx,clusterer,test_clusters,train_clusters)
    #print('The number of instances per group before target splitting:')
    #print(actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0])
    #print(inactives_train_idx[:10])

    #####
    ##Step 1.5: target split for inactives:
    #####
    #to do the target split, we can ignore the inactives_*_idx from above.
    possible_targets = [i for i in range(100)]
    #remove the active target:
    possible_targets.pop(idx)
    #now select some random test, and random train inactive targets:
    np.random.shuffle(possible_targets)
    inactive_test_targets = [possible_targets.pop() for _ in range(int(0.1*100))]
    inactive_train_targets = [possible_targets.pop() for _ in range(int(0.3*100))]
    #inactive_train_targets = possible_targets
    #now generate some new indices corresponding to these targets:
    inactives = ~(y_[:,idx].astype(bool))
    inactives_test_idx = np.logical_and(inactives, np.any(y_[:,inactive_test_targets],1)).nonzero()[0]
    inactives_train_idx =np.logical_and(inactives, np.any(y_[:,inactive_train_targets],1)).nonzero()[0]    

    print(actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0])
    if min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]])<20:        
           print('Not enough ligands to train and test')
           continue


       
    ######
    ###Trim inactives wrt both fingerprints:
    #####
    #trim from the inactives/train matrix first:
    morgan_to_trim = utils.trim(morgan_distance_matrix[inactives_test_idx], 
                                inactives_train_idx, 
                                inactives_test_idx,
                                fraction_to_trim=0.05, inverse=True)

    cats_to_trim = utils.trim(cats_distance_matrix[inactives_test_idx], 
                                inactives_train_idx, 
                                inactives_test_idx,
                                fraction_to_trim=0.05, inverse=True)

    inactives_to_trim = np.union1d(morgan_to_trim, cats_to_trim)
    new_inactives_train_idx = np.setdiff1d(inactives_train_idx, inactives_to_trim) #Return the unique values in ar1 that are not in ar2.

    
    ######
    ###Trim actives wrt both fingerprints:
    #####
    #trim from the inactives/train matrix first:
    morgan_to_trim = utils.trim(morgan_distance_matrix[inactives_test_idx], 
                                actives_train_idx, 
                                actives_test_idx,
                                fraction_to_trim=0.10, inverse=True)

    cats_to_trim = utils.trim(cats_distance_matrix[inactives_test_idx], 
                                actives_train_idx, 
                                actives_test_idx,
                                fraction_to_trim=0.10, inverse=True)

    actives_to_trim = np.union1d(morgan_to_trim, cats_to_trim)
    new_actives_train_idx = np.setdiff1d(actives_train_idx, actives_to_trim) #Return the unique values in ar1 that are not in ar2.


    print(f'Training data size: actives train: {len(new_actives_train_idx)}, inactives train: {len(new_inactives_train_idx)}')
    print(f'Testing data size: actives train: {len(actives_test_idx)}, inactives train: {len(inactives_test_idx)}')
    ######
    ###Evaluate the trimmed data:
    ######
    ave_morgan = utils.calc_AVE_quick(morgan_distance_matrix, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx)
    ave_cats = utils.calc_AVE_quick(cats_distance_matrix, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx)

    results_morgan = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=False)
    results_cats = utils.evaluate_split(catsMatrix_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=False)    
    df_after_trim.loc[loc_counter] = [ave_cats, ave_morgan, results_cats['ap'], results_morgan['ap']]


    #save data:
    df_after_trim.to_csv('./processed_data/graph_cluster_both/df_after_united_trim.csv')

    loc_counter += 1
