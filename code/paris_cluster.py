import numpy as np
from scipy import sparse
from pynndescent import NNDescent
from sknetwork.hierarchy import Paris
from sknetwork.hierarchy.postprocess import cut_straight

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
