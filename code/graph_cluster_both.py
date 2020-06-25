import utils
from os import path
import numpy as np
from scipy import stats, sparse
from paris_cluster import ParisClusterer
from sklearn.linear_model import LogisticRegression
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


clusterer = ParisClusterer(x_.toarray())
clusterer.loadAdjacency('./processed_data/distance_matrices/wadj_ecfp.npz')
clusterer.fit()


#Store all the results in these:
df_before_trim = pd.DataFrame(columns = ['ave_cats', 'ave_morgan', 'ap_cats', 'ap_morgan'])
df_after_morgan_trim = pd.DataFrame(columns = ['ave_cats', 'ave_morgan', 'ap_cats', 'ap_morgan'])
df_after_cats_trim = pd.DataFrame(columns = ['ave_cats', 'ave_morgan', 'ap_cats', 'ap_morgan'])


targets = list()
cutoffs = list()
aves = list()
sizes = list()

loc_counter = 0

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


    ######
    ###Evaluate the untrimmed data:
    ######
    results_morgan = utils.evaluate_split(x_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=False)
    results_cats = utils.evaluate_split(catsMatrix_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=False, ap=True, mcc=False)    
    ave_morgan= utils.calc_AVE_quick(morgan_distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    ave_cats= utils.calc_AVE_quick(cats_distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    df_before_trim.loc[loc_counter] = [ave_cats, ave_morgan, results_cats['ap'], results_morgan['ap']]


    ######
    ###Trim wrt ECFP
    #####
    #trim from the inactives/train matrix first:
    inactive_dmat = morgan_distance_matrix[inactives_test_idx]
    print('New inactives train_idx', inactive_dmat.shape, inactives_train_idx.shape, inactives_test_idx.shape)
    new_inactives_train_idx = utils.trim(inactive_dmat, 
                                       inactives_train_idx, 
                                       inactives_test_idx,
                                             fraction_to_trim=0.2)

    #then trim from the actives/train matrix:
    active_dmat = morgan_distance_matrix[actives_test_idx]
    print('New actives train_idx', active_dmat.shape, actives_train_idx.shape, actives_test_idx.shape)
    new_actives_train_idx = utils.trim(active_dmat,
                                    actives_train_idx, 
                                    actives_test_idx,
                                     fraction_to_trim=0.2)

    ######
    ###Evaluate the data trimmed wrt ECFP
    ######
    results_morgan = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True)
    results_cats = utils.evaluate_split(catsMatrix_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True)    
    ave_morgan= utils.calc_AVE_quick(morgan_distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    ave_cats= utils.calc_AVE_quick(cats_distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    df_after_morgan_trim.loc[loc_counter] = [ave_cats, ave_morgan, results_cats['ap'], results_morgan['ap']]


    ######
    ###Trim wrt CATS
    #####
    #trim from the inactives/train matrix first:
    inactive_dmat = cats_distance_matrix[inactives_test_idx]
    print('New inactives train_idx', inactive_dmat.shape, inactives_train_idx.shape, inactives_test_idx.shape)
    new_inactives_train_idx = utils.trim(inactive_dmat, 
                                       inactives_train_idx, 
                                       inactives_test_idx,
                                             fraction_to_trim=0.2)

    #then trim from the actives/train matrix:
    active_dmat = cats_distance_matrix[actives_test_idx]
    print('New actives train_idx', active_dmat.shape, actives_train_idx.shape, actives_test_idx.shape)
    new_actives_train_idx = utils.trim(active_dmat,
                                    actives_train_idx, 
                                    actives_test_idx,
                                     fraction_to_trim=0.2)


    ######
    ###Evaluate the data trimmed wrt CATS
    ######
    results_morgan = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True)
    results_cats = utils.evaluate_split(catsMatrix_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=False, ap=True)    
    ave_morgan= utils.calc_AVE_quick(morgan_distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    ave_cats= utils.calc_AVE_quick(cats_distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    df_after_cats_trim.loc[loc_counter] = [ave_cats, ave_morgan, results_cats['ap'], results_morgan['ap']]




    #save data:
    df_before_trim.to_csv('./processed_data/graph_cluster_both/df_before_trim.csv')
    df_after_morgan_trim.to_csv('./processed_data/graph_cluster_both/df_after_morgan_trim.csv')
    df_after_cats_trim.to_csv('./processed_data/graph_cluster_both/df_after_cats_trim.csv')

    loc_counter += 1
