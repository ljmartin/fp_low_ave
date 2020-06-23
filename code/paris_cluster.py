import numpy as np
from scipy import sparse
from pynndescent import NNDescent
from sknetwork.hierarchy import Paris
from sknetwork.hierarchy.postprocess import cut_straight

from tqdm import tqdm

class ParisClusterer(object):
    def __init__(self, featureMatrix, metric):
        self.featureMatrix = featureMatrix
        self.metric = metric

    def loadAdjacency(self, name):
        self.wdAdj = sparse.load_npz(name)
        
    def fit(self):
        self.paris = Paris()
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
