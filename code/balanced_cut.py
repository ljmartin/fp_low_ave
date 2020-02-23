import numpy as np


def balanced_cut(dendrogram, max_cluster_size):
    n_nodes = dendrogram.shape[0] + 1
    labels = np.zeros(n_nodes, dtype=int)
    cluster = {node: [node] for node in range(n_nodes)}
    completed_clusters = list()
    
    for t in range(n_nodes - 1):
        currentID = n_nodes+t
        left = cluster[int(dendrogram[t][0])]
        right = cluster[int(dendrogram[t][1])]
        if len(left)+len(right) > max_cluster_size:
            for clust in [left, right]:
                if len(clust)<max_cluster_size:
                    completed_clusters.append(clust)
                    
        cluster[currentID] = cluster.pop(int(dendrogram[t][0])) + cluster.pop(int(dendrogram[t][1]))
    
    for count, indices in enumerate(completed_clusters):
        labels[indices]=count
    return labels
