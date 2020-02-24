import utils
from os import path

import numpy as np
from scipy import stats, sparse

from pynndescent import NNDescent

from tqdm import tqdm

import statsmodels.stats.api as sms

##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())

#Fingerprints to be compared:
fp_names = utils.getNames()
fp_dict = {}
fp_similarities = {}


#split:
idx = np.random.choice(252409, 1000, replace=False)
mask = np.ones(252409, dtype=bool)
mask[idx]=False
    
for fp in fp_names:
    print('Loading:', fp)
    x_, y = utils.load_feature_and_label_matrices(type=fp)
    print('Analysing:', fp, 'shape:', x_.shape[0])

    print('Splitting:')
    train_x, train_y = x_[mask], y[mask]
    test_x, test_y = x_[~mask], y[~mask]

    print('Buildig NN tree:')
    nn_index = NNDescent(train_x.toarray(), n_neighbors=101, metric='cosine')
    print('Getting neighbour graph')
    train_n, train_d = nn_index.query(train_x.toarray())
    print('Querying with test ligands:')
    test_n, test_d = nn_index.query(test_x.toarray())

    print('Calculating shared nearest neighbours:')
    for test_neighbour in tqdm(test_n):
        distances = 1-np.array([(test_neighbour[:100]==train_neighbour[1:101])/100 for train_neighbour in train_n])
        true_labels = train_y[:,train_y[idx].nonzero()[0][0]]
        roc = np.cumsum(true_labels[distances.argsort()])/sum(true_labels)
        scores.append(roc)
    scores = np.array(scores)
    low, high = sms.DescrStatsW(scores).tconfint_mean()
    np.save('./processed_data/snn/'+fp+'_roc.npy', scores.mean(0))
    np.save('./processed_data/snn/'+fp+'_low.npy', scores.std(0))
    np.save('./processed_data/snn/'+fp+'_high.npy', scores.std(0))
            
