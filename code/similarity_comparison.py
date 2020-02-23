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


##The J&J benchmark used single-linkage clustering. 
#Wallach et. al later showed how model performance was related to bias. 
#Here we reproduce this using a set of train/test splits
#also using single-linkage clustering.

#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')

#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {}
fp_similarities = {}

#Load up the dictionaries with the relevant feature matrices for each fingerprint:
for fp in fp_names:
    print('Loading:', fp)
    featureMatrix, labels = utils.load_feature_and_label_matrices(type=fp)
    fp_dict[fp]=sparse.csr_matrix(featureMatrix)
    
for fp in fp_names:
    print('Analysing:', fp)
    x_ = fp_dict[fp]
    if x_.dtype==int:
        scores = list()
        for _ in tqdm(range(1000)):
            idx = np.random.choice(x_.shape[0])
            lig = x_[idx]
            distances = utils.fast_jaccard(lig, x_)
            true_labels = y[:,y[idx].nonzero()[0][0]]
            roc = np.cumsum(true_labels[distances.argsort()])/sum(true_labels)
            scores.append(roc)
        scores = np.array(scores)
        np.save('./processed_data/similarities/'+fp+'_roc.npy', scores.mean(0))
        np.save('./processed_data/similarities/'+fp+'_stderr.npy', scores.std(0))
            
