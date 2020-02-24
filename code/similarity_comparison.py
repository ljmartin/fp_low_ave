import utils
from os import path

import numpy as np
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform, cdist

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import statsmodels.stats.api as sms

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {}
fp_similarities = {}

    
for fp in fp_names:
    print('Loading:', fp)
    x_, y = utils.load_feature_and_label_matrices(type=fp)
    print('Analysing:', fp)
    
    scores = list()
    for _ in tqdm(range(1000)):
        idx = np.random.choice(x_.shape[0])
        lig = x_[idx]
        if x_.dtype==int:
            distances = utils.fast_jaccard(lig, x_)
        else:
            distances = 1-cosine_similarity(lig, x_)
        true_labels = y[:,y[idx].nonzero()[0][0]]
        roc = np.cumsum(true_labels[distances.argsort()])/sum(true_labels)
        scores.append(roc)
    scores = np.array(scores)
    low, high = sms.DescrStatsW(scores).tconfint_mean()
    np.save('./processed_data/similarities/'+fp+'_roc.npy', scores.mean(0))
    np.save('./processed_data/similarities/'+fp+'_low.npy', scores.std(0))
    np.save('./processed_data/similarities/'+fp+'_high.npy', scores.std(0))
            
