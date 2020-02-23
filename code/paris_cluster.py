import numpy as np
from scipy import sparse
from pynndescent import NNDescent
from sknetwork.hierarchy import Paris
from sknetwork.hierarchy import straight_cut

from tqdm import tqdm

class ParisClusterer(object):
    def __init__(self, featureMatrix):
        self.featureMatrix = featureMatrix

    def buildAdjacency(self, type='pynndescent', nn=50, metric='dice'):
        print('Building nearest neighbor graph (slowest step)...')
        if type=='pynndescent':
            nn_index = NNDescent(self.featureMatrix, n_neighbors=nn, metric=metric)
            n, d = nn_index.neighbor_graph
            self.n = n
            self.d = d
        print('Done')
        print('Building weighted, directed adjacency matrix...')
        wdAdj = sparse.dok_matrix((self.featureMatrix.shape[0], self.featureMatrix.shape[0]), dtype=float)
        for neighbours, distances in tqdm(zip(n, d)):
            instanceIndex = neighbours[0]
            for neighbourIndex, distance in zip(neighbours[1:], distances[1:]):
                wdAdj[instanceIndex, neighbourIndex] += 1-distance #similarity = 1-distance
        self.wdAdj = sparse.csr_matrix(wdAdj).astype(float)
        
        
    def fit(self):
        self.paris = Paris(engine='numba')
        self.paris.fit(self.wdAdj)

    def balanced_cut(self, max_cluster_size):
        n_nodes = self.paris.dendrogram_.shape[0] + 1
        labels = np.zeros(n_nodes, dtype=int)
        cluster = {node: [node] for node in range(n_nodes)}
        completed_clusters = list()
    
        for t in range(n_nodes - 1):
            currentID = n_nodes+t
            left = cluster[int(self.paris.dendrogram_[t][0])]
            right = cluster[int(self.paris.dendrogram_[t][1])]
            if len(left)+len(right) > max_cluster_size:
                for clust in [left, right]:
                    if len(clust)<max_cluster_size:
                        completed_clusters.append(clust)
                    
            cluster[currentID] = cluster.pop(int(self.paris.dendrogram_[t][0])) + cluster.pop(int(self.paris.dendrogram_[t][1]))
    
        for count, indices in enumerate(completed_clusters):
            labels[indices]=count

        self.labels_ = labels
        
#    def balancedCut(self, maxClusterSize):
#        #self.labels_, _ = ward_cut_tree_balanced(self.paris.dendrogram_, clusterSize)
#        dendrogram = self.paris.dendrogram_.copy()
#        labels = np.zeros(len(dendrogram)+1).astype(int)-1
#        mask = np.ones(len(dendrogram)+1).astype(bool)
#        last_cluster_id = 0
#        
#        for n_clusters in range(1, len(dendrogram)+1):
#            temp_labels = straight_cut(dendrogram, n_clusters=n_clusters)
#            ids, counts = np.unique(temp_labels[mask], return_counts=True)
#            if min(counts)<maxClusterSize: #do we at have at least one group being under the min cluster size?
#                smaller_than_max = ids[counts<maxClusterSize] #if yes, then get those groups.
#                for temp_id in smaller_than_max: #There might be multiple. For each group, set the 
#                				 #instance labels to the queued cluster ID
#                    labels[temp_labels==temp_id]=last_cluster_id
#                    mask[temp_labels==temp_id]=False
#                    last_cluster_id+=1
#            if min(labels)!=-1:
#                break
#        self.labels_ = labels
    
