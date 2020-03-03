import utils
from os import path

import numpy as np
from scipy import stats, sparse

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
import matplotlib.pyplot as plt

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())
utils.set_mpl_params()

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(243, 10, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)


#This will be used for clustering:
distance_matrix = utils.fast_dice(x_)


#choose a random target:
idx = np.random.choice(y_.shape[1])
all_positive_indices = (y_[:,idx]==1).nonzero()[0]
pos_test_counts = {index: 0 for index in all_positive_indices}

all_negative_indices = (y_[:,idx]==0).nonzero()[0]
neg_test_counts = {index: 0 for index in all_negative_indices}

positive_fractions = []
negative_fractions = []

for idx in range(10):
    all_positive_indices = (y_[:,idx]==1).nonzero()[0]
    pos_test_counts = {index: 0 for index in all_positive_indices}

    all_negative_indices = (y_[:,idx]==0).nonzero()[0]
    neg_test_counts = {index: 0 for index in all_negative_indices}
    
    for _ in tqdm(range(30)):
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


        for i in actives_test_idx:
            pos_test_counts[i] +=1
        for i in inactives_test_idx:
            neg_test_counts[i] +=1


    pos_frac = np.array(list(pos_test_counts.values()))/30
    neg_frac = np.array(list(neg_test_counts.values()))/30
    
    positive_fractions.append(pos_frac)
    negative_fractions.append(neg_frac)


fig, ax = plt.subplots()
xr = np.linspace(0,1,200)
count=0

for a,b in zip(positive_fractions, negative_fractions):
    dens = sm.nonparametric.KDEUnivariate(a)
    dens.fit(bw=0.0175)
    if count==9:
        l1 = 'Active test fraction'
        l2 = 'Inactive test fraction'
    else:
        l1 = ""
        l2 = ""
    ax.plot([0]+list(xr), [0]+list(dens.evaluate(xr)), c='C0', lw=2, label=l1)
    dens = sm.nonparametric.KDEUnivariate(b)
    dens.fit(bw=0.0175)
    ax.plot([0]+list(xr), [0]+list(dens.evaluate(xr)), c='C1', lw=2, label=l2)
    count+=1
    
ax.grid()
ax.set_xlabel('% in test set')
ax.set_ylabel('Density')
ax.axvline(0.2, linestyle='--', c='k', label='Goal value')
ax.legend()
fig.savefig('./processed_data/supplementary/test_set_occupancy.png')
