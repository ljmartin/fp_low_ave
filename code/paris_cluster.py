import numpy as np
from scipy import sparse
from pynndescent import NNDescent
from sknetwork.hierarchy import Paris
from balanced_cut import ward_cut_tree_balanced

class ParisClusterer(object):
    def __init__(self, featureMatrix):
        self.featureMatrix = featureMatrix

    def buildAdjacency(self, type='pynndescent', nn=50, metric='dice'):
        print('Building nearest neighbor graph...')
        if type=='pynndescent':
            nn_index = NNDescent(self.featureMatrix, n_neighbors=nn, metric=metric)
            n, d = nn_index.neighbor_graph
        print('Done')
        print('Building weighted, directed adjacency matrix...')
        wdAdj = sparse.dok_matrix((self.featureMatrix.shape[0], self.featureMatrix.shape[0]), dtype=float)
        for neighbours, distances in tqdm(zip(n, d)):
            instanceIndex = neighbours[0]
            for neighbourIndex, distance in zip(neighbours[1:], distances[1:]):
                wdAdj[instanceIndex, neighbourIndex] += 1-distance #similarity = 1-distance
        self.wdAdj = sparse.csr_matrix(wdAdj).atype(float)
        
        
    def fit(self):
        self.paris = Paris(engine='numba')
        self.paris.fit(self.wdAdj)

    def balancedCut(self, clusterSize):
        self.labels_, _ = ward_cut_tree_balanced(self.paris.dendrogram_, clusterSize)
        
