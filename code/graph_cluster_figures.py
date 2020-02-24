import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

##Set plotting parameters:
utils.set_mpl_params()


aves = np.load('processed_data/graph_cluster/aves.npy', allow_pickle=True)
targets = np.load('processed_data/graph_cluster/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_cluster/cutoffs.npy', allow_pickle=True)
sizes = np.load('processed_data/graph_cluster/sizes.npy', allow_pickle=True)


fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(6)

ax.scatter(cutoffs, aves, c=targets, alpha=utils.ALPHA)
ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C1', label='Target 1')
ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C2', label='Target 2')
ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C3', label='etc...')
ax.set_xlabel('Cutoff')
ax.set_ylabel('AVE score')
ax.legend()
ax.grid()

fig.savefig('./processed_data/graph_cluster/cutoff_vs_ave.png')
plt.close(fig)

