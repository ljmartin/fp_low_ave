import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot
import matplotlib.patheffects as mpe

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

##Set plotting parameters:
utils.set_mpl_params()

aves_before_trim = np.load('./processed_data/graph_cluster/aves_before_trim.npy')
aves_after_trim = np.load('./processed_data/graph_cluster/aves_after_trim.npy')
ap_before_trim = np.load('./processed_data/graph_cluster/ap_before_trim.npy')
ap_after_trim = np.load('./processed_data/graph_cluster/ap_after_trim.npy')

targets = np.load('processed_data/graph_cluster/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_cluster/cutoffs.npy', allow_pickle=True)
#sizes = np.load('processed_data/graph_cluster/sizes.npy', allow_pickle=True)


fig, ax = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)

kdeplot(aves_before_trim, ax=ax[0], label='Before trim')
kdeplot(aves_after_trim, ax=ax[0], label='After trim')
ax[0].set_ylabel('Density')
ax[0].set_xlabel('AVE')
ax[0].grid()
ax[0].legend()
utils.plot_fig_label(ax[0], 'A.')


ax[1].scatter(aves_before_trim, ap_before_trim, alpha=utils.ALPHA, label='Before trim')
ax[1].scatter(aves_after_trim, ap_after_trim, alpha=utils.ALPHA, label='After trim')
ax[1].set_xlabel('AVE')
ax[1].set_ylabel('AP')
ax[1].legend()
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')

fig.savefig('./processed_data/graph_cluster/trim.png')
plt.close(fig)



