import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, cdist, squareform
import copy

from pynndescent import NNDescent
from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_loss
from sklearn.metrics import confusion_matrix, average_precision_score, label_ranking_average_precision_score

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def getSeed(seed=500):
    return seed

def load_feature_and_label_matrices(type='ecfp'):
    y = sparse.load_npz('./raw_data/y.npz').toarray()
    if type=='ecfp':
        x = sparse.load_npz('./processed_data/fingerprints/morgan.npz').toarray()
    return x, y


def get_subset(x, y, indices):
    y_ = y[:,indices]
    #remove ligands that do not have a positive label in the subset
    row_mask = y_.sum(axis=1)>0
    y_ = y_[row_mask]
    x_ = x[row_mask]
    return x_, y_
    

def merge_feature_matrices(matrices):
    """Merges four feature matrices into two matrices (test/train) for subsequent
    model fitting by sklearn. It also generates label matrices. 
    The matrices are 2-dimensional feature matrices, often called "X" in sklearn
    terminology, i.e. shape (N,F) where N is the number of instances and F is 
    the feature dimension. 
    
    Parameters:
    	matrices (list): four matrices of 2D numpy arrays. These are:
    	- x_actives_train: Positive instances for the training set
    	- x_actives_test: Positive instances for the test set
    	- x_inactives_train: Negative instances for the training set
    	- x_inactives_test: Negative instances for the test set

    Returns:
    	- x_train, y_train, x_test, y_test: feature and label matrices for 
    	an sklearn classifier i.e. clf.fit(x_train, y_train), or 
    	clf.score(x_test, y_test).
    """
    x_actives_train, x_actives_test, x_inactives_train, x_inactives_test = matrices

    x_train = np.vstack([x_actives_train, x_inactives_train]) #stack train instances together
    x_test = np.vstack([x_actives_test, x_inactives_test]) #stack test instance together
    #build 1D label array based on the sizes of the active/inactive train/test 
    y_train = np.zeros(len(x_train))
    y_train[:len(x_actives_train)]=1
    y_test = np.zeros(len(x_test))
    y_test[:len(x_actives_test)]=1    
    return x_train, x_test, y_train, y_test

def split_feature_matrices(x_train, x_test, y_train, y_test, idx):
    """Does the opposite of merge_feature_matrices i.e. when given the 
    train and test matrices for features and labels, splits them into 
    train/active, test/active, train/inactive, test/inactive.

    Parameters:
    	- x_train, x_test, y_train, y_test (2D np.arrays): Feature 
        matrices and label matrices in the sklearn 'X' and 'Y' style.
        - idx (int): a column of the label matrix corresponding to the 
        protein target you wish to test. """
    x_actives_train = x_train[y_train[:,idx]==1]
    x_actives_test = x_test[y_test[:,idx]==1]
    
    x_inactives_train = x_train[y_train[:,idx]!=1]
    x_inactives_test = x_test[y_test[:,idx]!=1]
    
    return x_actives_train, x_actives_test, x_inactives_train, x_inactives_test


##The following performs test/train splitting by single-linkage clustering:
def make_cluster_split(x_, y_, clust, percentage_holdout=0.2, test_clusters=False):
    """Given a X,Y, and a fitted clusterer from sklearn, this selects
    a percentage of clusters as holdout clusters, then constructs the X,Y matrices

    Parameters:
    	x_ (2d np.array): Feature matrix X
        y_ (2d np.array): Label matrix Y
        percentage_hold_out (float): Percentage of ligands desired as hold-out data.

    Returns:
    	x_train, x_test, y_train, y_test: feature and label matrices
        for an sklearn classifier. If this is confusing, see sklearn's 
        train_test_split function."""
    if isinstance(test_clusters, bool):
        test_clusters = np.random.choice(clust.labels_.max(), int(clust.labels_.max()*percentage_holdout), replace=False)
    mask = ~np.isin(clust.labels_, test_clusters)
    x_test = x_[~mask]
    x_train = x_[mask]
    y_test = y_[~mask]
    y_train = y_[mask]
    return x_train, x_test, y_train, y_test

##The following is to calculate AVE bias:
def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A

def fast_dice(X, Y=None):
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X).astype(bool).astype(int)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y).astype(bool).astype(int)
            
    intersect = X.dot(Y.T)
    #cardinality = X.sum(1).A
    cardinality_X = X.getnnz(1)[:,None] #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    cardinality_Y = Y.getnnz(1) #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    return (1-(2*intersect) / (cardinality_X+cardinality_Y.T)).A

def calcDistMat( fp1, fp2, metric='jaccard' ):
    """Calculates the pairwise distance matrix between features
    fp1 and fp2"""
    return cdist(fp1, fp2, metric=metric)



def calc_distance_matrices(matrices, metric='dice'):
    """Performs the first step in calculating AVE bias: 
    calculating distance matrix between each pair of ligand sets. 
    
    Parameters:
    	matrices (list): set of four feature matrices in the order:
        x_actives_train, x_actives_test, x_inactives_train, x_inactives_test

    Returns:
    	distances (list): set of four distance matrices. Columns will
        be equal to the number of test set ligands, rows will be equal to 
        the number of train set ligands. """

    x_actives_train, x_actives_test, x_inactives_train, x_inactives_test = matrices
    #original method (slow - do not use):
    #aTest_aTrain_D = calcDistMat( x_actives_test, x_actives_train, metric )
    #aTest_iTrain_D = calcDistMat( x_actives_test, x_inactives_train, metric )
    #iTest_iTrain_D = calcDistMat( x_inactives_test, x_inactives_train, metric )
    #iTest_aTrain_D = calcDistMat( x_inactives_test, x_actives_train, metric )

    if metric=='jaccard':
        distFun = fast_jaccard
    if metric=='dice':
        distFun = fast_dice
    
    #faster using sparse input data to avoid calculating lots of zeroes:
    aTest_aTrain_D = distFun(x_actives_test, x_actives_train)
    aTest_iTrain_D = distFun(x_actives_test, x_inactives_train)
    iTest_iTrain_D = distFun(x_inactives_test, x_inactives_train)
    iTest_aTrain_D = distFun(x_inactives_test, x_actives_train)
    return aTest_aTrain_D, aTest_iTrain_D, iTest_iTrain_D, iTest_aTrain_D
    
def calc_AVE(distances):
    """Calculates the AVE bias. Please see Wallach et.al https://doi.org/10.1021/acs.jcim.7b00403

    Parameters:
	distances (list): list of distances matrices returned by 
        `calc_distance_matrices()`
        
    Returns:
    	AVE (float): the AVE bias"""

    aTest_aTrain_D, aTest_iTrain_D, iTest_iTrain_D, iTest_aTrain_D = distances
    
    aTest_aTrain_S = np.mean( [ np.mean( np.any( aTest_aTrain_D < t, axis=1 ) ) for t in np.linspace( 0, 1.0, 50 ) ] )
    aTest_iTrain_S = np.mean( [ np.mean( np.any( aTest_iTrain_D < t, axis=1 ) ) for t in np.linspace( 0, 1.0, 50 ) ] )
    iTest_iTrain_S = np.mean( [ np.mean( np.any( iTest_iTrain_D < t, axis=1 ) ) for t in np.linspace( 0, 1.0, 50 ) ] )
    iTest_aTrain_S = np.mean( [ np.mean( np.any( iTest_aTrain_D < t, axis=1 ) ) for t in np.linspace( 0, 1.0, 50 ) ] )
    
    AVE = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
    return AVE

def calc_VE(distances):
    """Calculate the VE bias score. Please see Davis et. al at DOI 2001.03207 
    pre-print is available at: https://arxiv.org/abs/2001.03207 

    Parameters:
        distances (list): list of distances matrices returned by
        `calc_distance_matrices()`

    Returns:
        VE: the VE bias"""
    
    aTest_aTrain_D, aTest_iTrain_D, iTest_iTrain_D, iTest_aTrain_D = distances
    term_one = np.mean(aTest_iTrain_D.min(axis=1) - aTest_aTrain_D.min(axis=1))
    term_two = np.mean(iTest_aTrain_D.min(axis=1) - iTest_iTrain_D.min(axis=1))
    VE = np.sqrt(term_one**2+term_two**2)
    return VE


##For plotting in a particular style: 
##Please see https://github.com/ColCarroll/minimc for the source of inspiration
ALPHA = 0.7
def plot_fig_label(ax, lab):
    ax.text(-0.1, 1.15, lab, transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='right')

def set_mpl_params():
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#1b6989",
                    "#e69f00",
                    "#009e73",
                    "#f0e442",
                    "#50b4e9",
                    "#d55e00",
                    "#cc79a7",
                    "#000000",
                ],
            ),
            "scatter.edgecolors": 'k',
            "grid.linestyle": '--',
            "font.serif": [
                "Palatino",
                "Palatino Linotype",
                "Palatino LT STD",
                "Book Antiqua",
                "Georgia",
                "DejaVu Serif",
                ],
            "font.family": "serif",
            "figure.facecolor": "#fffff8",
            "axes.facecolor": "#fffff8",
            "axes.axisbelow": True,
            "figure.constrained_layout.use": True,
            "font.size": 14.0,
            "hist.bins": "auto",
            "lines.linewidth": 3.0,
            "lines.markeredgewidth": 2.0,
            "lines.markerfacecolor": "none",
            "lines.markersize": 8.0,
        }

    )
