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
x_100, y_100 = utils.get_subset(x, y, indices=col_indices)

#Open a memory mapped distance matrix.
#We do this because the pairwise distance matrix for 100 targets does not fit in memory.
#It is nearly 100% dense and has 117747*117747 = 13864356009 elements. This is also
#Why it uses float16 (reducing the required storage space to ~26GB, c.f. 52GB for float32).
distance_matrix = np.memmap('./processed_data/distance_matrices/morgan_distance_matrix.dat', dtype=np.float16,
                            shape=(x_100.shape[0], x_100.shape[0]))


#for replicating the AVE, we will use a small dataset of 10 targets.
col_indices = np.random.choice(100, 10, replace=False)
x_, y_ = utils.get_subset(x_100, y_100, indices=col_indices)
#now cut the distance_matrix down to size:
temp_ = y_100[:,col_indices]
row_mask = temp_.sum(axis=1)>0
distance_matrix = distance_matrix[row_mask][:,row_mask]


#this sparse matrix is the adjacency graph
#using DOK matrix because it's faster to write. It is converted to CSR after.
wdAdj = sparse.dok_matrix((x_.shape[0], x_.shape[0]), dtype=float)
#iterate through every row, writing adjacencies for the 50-NN.  
for row in tqdm(range(x_.shape[0])):
    all_distances = distance_matrix[row]
    neighbors = np.argpartition(all_distances, 50)[:50]
    distances = all_distances[neighbors]
    for neighbourIndex, distance in zip(neighbors[1:], distances[1:]):
        wdAdj[row, neighbourIndex] += 1-distance # because similarity is 1-distance, and this a weighted adjacency

sparse.save_npz('./processed_data/distance_matrices/small_wadj_ecfp.npz', sparse.csr_matrix(wdAdj))

clusterer = ParisClusterer(x_.toarray())
clusterer.loadAdjacency('./processed_data/distance_matrices/small_wadj_ecfp.npz')
clusterer.fit()




import pandas as pd
df = pd.DataFrame(columns=['ave', 'ap',
                           'mcc', 
                           'ef', 'auroc'])



count = 0 

for _ in tqdm(range(400)):
    #choose a random target:
    idx = np.random.choice(y_.shape[1])

    #choose a random cluster size upper limit and cluster:
    clusterSize = np.random.randint(25,200)
    clusterer.labels_ = utils.cut_balanced(clusterer.paris.dendrogram_, clusterSize)

    clabels = np.unique(clusterer.labels_)
    pos_labels = np.unique(clusterer.labels_[y_[:,idx]==1])
    neg_labels = clabels[~np.isin(clabels, pos_labels)]
    if min(len(pos_labels), len(neg_labels))<2:
        print('Not enough positive clusters to split')
        continue

    #test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.1, [0.1,0.1], shuffle=True)
    test_clusters, train_clusters = utils.split_clusters(pos_labels, neg_labels, 0.2, 0.2, shuffle=True)

    actives_test_idx, actives_train_idx, inactives_test_idx, inactives_train_idx = utils.get_four_matrices(y_,idx,clusterer,test_clusters,train_clusters)
    print(f'ActivesTest: {actives_test_idx.shape[0]},\nActivesTrain: {actives_train_idx.shape[0]}')
    print(f'InactivesTest: {inactives_test_idx.shape[0]},\nInactivesTrain: {inactives_train_idx.shape[0]}')
    if min([actives_test_idx.shape[0], actives_train_idx.shape[0], inactives_test_idx.shape[0], inactives_train_idx.shape[0]])<20:        
           print('Not enough ligands to train and test')
           continue
    ave = utils.calc_AVE_quick(distance_matrix, actives_train_idx, actives_test_idx,inactives_train_idx, inactives_test_idx)
    

    #evaluate a LogReg model using the original single-linkage split
    results = utils.evaluate_split(x_, y_, idx, actives_train_idx, actives_test_idx, inactives_train_idx, inactives_test_idx, auroc=True, ap=True, mcc=True, ef=True)

    df.loc[count] = [ave, 
                     results['ap'],
                     results['mcc'],
                     results['ef'],
                     results['auroc']]
    count+=1


    df.to_csv('./processed_data/replicate_AVE/results.csv')
