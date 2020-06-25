import utils
from os import path
import pandas as pd
import numpy as np
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics.pairwise import cosine_distances
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
col_indices = np.random.choice(226, 6, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)

#load cats one as well:
x_cats, y = utils.load_feature_and_label_matrices(type='cats')
x_cats_, _ = utils.get_subset(x_cats, y, indices=col_indices)

#This will be used for clustering:
distance_matrix = utils.fast_dice(x_)
cats_distance_matrix = (cosine_distances(x_cats_)/2)**0.25


#These will be used to save all the data so we don't have to repeatedly run this script
targets = list() #store which target is analyzed (not used)
results_df = pd.DataFrame(columns = ['ave_cats_before', 'ave_cats_after', 'ave_morgan_before',
                                     'ave_morgan_after', 'auroc_before', 'auroc_after',
                                     'ap_before', 'ap_after', 'mcc_before', 'mcc_after',
                                     'ef_before', 'ef_after', 'targets'])

loc_counter=0

for _ in tqdm(range(200)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random clustering cutoff and cluster:
    cutoff = stats.uniform(0.05, 0.425).rvs()
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage='single', affinity='precomputed')
    clusterer.fit(distance_matrix)

    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
    neg_labels = clabels[~np.isin(clabels, pos_labels)]
    if min(len(pos_labels), len(neg_labels))<2:
        #print('Not enough positive clusters to split')
        continue
    
    test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.2, 0.2, shuffle=True)

    actives_test_idx, actives_train_idx, inactives_test_idx, inactives_train_idx = utils.get_four_matrices(y_,idx,clusterer,test_clusters,train_clusters)
    if min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]])<20:        
        #print('Not enough ligands to train and test')
        continue

    results_dict = dict()

    #####
    ##Calculate AVE's wrt each fingerprint, before trimming nearest neighbors.
    #####    
    ave_morgan= utils.calc_AVE_quick(distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    ave_cats= utils.calc_AVE_quick(cats_distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    results_dict['ave_cats_before']=ave_cats
    results_dict['ave_morgan_before']=ave_morgan
    
    #evaluate a LogReg model using the original single-linkage split
    results = utils.evaluate_split(x_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=True, ap=True, mcc=True, ef=True)
    results_dict['auroc_before']=results['auroc']
    results_dict['ap_before']=results['ap']
    results_dict['mcc_before']=results['mcc']
    results_dict['ef_before']=results['ef']



    #####
    ##Trim some nearest neighbors from the training set, then repeat:
    #####
    
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

    ave_morgan= utils.calc_AVE_quick(distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    ave_cats= utils.calc_AVE_quick(cats_distance_matrix, new_actives_train_idx, actives_test_idx,new_inactives_train_idx, inactives_test_idx)
    results_dict['ave_cats_after']=ave_cats
    results_dict['ave_morgan_after']=ave_morgan

    #evaluate a LogReg model using the original single-linkage split
    results = utils.evaluate_split(x_, y_, idx, new_actives_train_idx, actives_test_idx, new_inactives_train_idx, inactives_test_idx, auroc=True, ap=True, mcc=True, ef=True)
    results_dict['auroc_after']=results['auroc']
    results_dict['ap_after']=results['ap']
    results_dict['mcc_after']=results['mcc']
    results_dict['ef_after']=results['ef']

    results_dict['targets']=idx

    results_df.loc[loc_counter] = results_dict
    loc_counter+=1
    

    results_df.to_csv('./processed_data/replicate_AVE/results_df.csv')
