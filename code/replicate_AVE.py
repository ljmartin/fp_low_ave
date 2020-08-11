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
col_indices = np.random.choice(226, 10, replace=False)
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



import pandas as pd
df = pd.DataFrame(columns=['ave_before_trim', 'ave_after_trim',
                           'ap_before_trim', 'ap_after_trim',
                           'mcc_before_trim', 'mcc_after_trim',
                           'ef_before_trim', 'ef_after_trim',
                           'cluster_size', 'active_test_fraction', 'inactive_test_fraction'])


count = 0 

for _ in tqdm(range(100)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random cluster size upper limit and cluster:
    clusterSize = np.random.randint(50,250)
    clusterer.labels_ = utils.cut_balanced(clusterer.paris.dendrogram_, clusterSize)

    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
    neg_labels = clabels[~np.isin(clabels, pos_labels)]
    if min(len(pos_labels), len(neg_labels))<2:
        print('Not enough positive clusters to split')
        continue

    #test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.1, [0.1,0.1], shuffle=True)
    test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.2, [0.1, 0.3], shuffle=True)

    actives_test_idx, actives_train_idx, inactives_test_idx, inactives_train_idx = utils.get_four_matrices(y_,idx,clusterer,test_clusters,train_clusters)
    print(f'ActivesTest: {actives_test_idx.shape[0]},\nActivesTrain: {actives_train_idx.shape[0]}')
    print(f'InactivesTest: {inactives_test_idx.shape[0]},\nInactivesTrain: {inactives_train_idx.shape[0]}')
    if min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]])<20:        
           print('Not enough ligands to train and test')
           continue
    ave_before_trim = utils.calc_AVE_quick(distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    
    #Now we will trim some nearest neighbours and by doing so, reduce AVE.
    #trim from the inactives/train matrix first:
    inactive_dmat = distance_matrix[inactives_test_idx]
    new_inactives_train_idx = utils.trim(inactive_dmat, 
                                       inactives_train_idx, 
                                       inactives_test_idx,
                                             fraction_to_trim=0.2)

    #then trim from the actives/train matrix:
    active_dmat = distance_matrix[actives_test_idx]
    new_actives_train_idx = utils.trim(active_dmat,
                                    actives_train_idx, 
                                    actives_test_idx,
                                       fraction_to_trim=0.2)


    print('###After trimming###')
    print(f'ActivesTest: {actives_test_idx.shape[0]},\nActivesTrain: {new_actives_train_idx.shape[0]}')
    print(f'InactivesTest: {inactives_test_idx.shape[0]},\nInactivesTrain: {new_inactives_train_idx.shape[0]}')
    #now calculate AVE with this new split:
    ave_after_trim = utils.calc_AVE_quick(distance_matrix, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx)
 

    #evaluate a LogReg model using the original single-linkage split
    results_before_trim = utils.evaluate_split(x_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=True, ef=True)

    #evaluate a LogReg model using the new (lower AVE) split:
    results_after_trim = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=True, ef=True)


    va = actives_test_idx.shape[0]
    ta = new_actives_train_idx.shape[0]
    vi = inactives_test_idx.shape[0]
    ti = new_inactives_train_idx.shape[0]
    
    df.loc[count] = [ave_before_trim, ave_after_trim,
                     results_before_trim['ap'], results_after_trim['ap'],
                     results_before_trim['mcc'], results_after_trim['mcc'],
                     results_before_trim['ef'], results_after_trim['ef'],
                     clusterSize, va/(va+ta), vi/(vi+ti)]
    count+=1


    df.to_csv('./processed_data/replicate_AVE/results.csv')
