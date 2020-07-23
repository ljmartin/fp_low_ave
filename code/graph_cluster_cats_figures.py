import numpy as np
import matplotlib.pyplot as plt
from seaborn import kdeplot
import matplotlib.patheffects as mpe

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

from scipy.stats import norm

from tqdm import tqdm

##Set plotting parameters:
utils.set_mpl_params()

#aves_before_trim = np.load('./processed_data/graph_cluster/aves_before_trim.npy')
#aves_after_trim = np.load('./processed_data/graph_cluster/aves_after_trim.npy')
#ap_before_trim = np.load('./processed_data/graph_cluster/ap_before_trim.npy')
#ap_after_trim = np.load('./processed_data/graph_cluster/ap_after_trim.npy')

#targets = np.load('processed_data/graph_cluster/targets.npy', allow_pickle=True)
#cutoffs = np.load('processed_data/graph_cluster/cutoffs.npy', allow_pickle=True)
#sizes = np.load('processed_data/graph_cluster/sizes.npy', allow_pickle=True)


import pandas as pd
df = pd.read_csv('./processed_data/graph_cluster_cats/results.csv')
aves_before_trim = df['ave_before_trim']
aves_after_trim = df['ave_after_trim']
ap_before_trim = df['ap_before_trim']
ap_after_trim = df['ap_after_trim']

fig, ax = plt.subplots(1,2)
fig.set_figheight(5)
fig.set_figwidth(10)

mu_before, sigma_before = norm.fit(aves_before_trim)
mu_after, sigma_after = norm.fit(aves_after_trim)

kdeplot(aves_before_trim, ax=ax[0], label=f'Before trim, \nμ={np.around(mu_before,3)}, σ={np.around(sigma_before,3)}')
kdeplot(aves_after_trim, ax=ax[0], label=f'After trim, \nμ={np.around(mu_after, 3)}, σ={np.around(sigma_after,3)}')
ax[0].set_ylabel('Density')
ax[0].set_xlabel('AVE')
ax[0].grid()
ax[0].legend()
utils.plot_fig_label(ax[0], 'A.')


#for a,b,c,d in zip(aves_before_trim, aves_after_trim, ap_before_trim, ap_after_trim):
#    ax[1].plot([a,b], [c,d], lw=0.2, c='k')
ax[1].scatter(aves_before_trim, ap_before_trim, label='Before trim')
ax[1].scatter(aves_after_trim, ap_after_trim, label='After trim')
ax[1].set_xlabel('AVE')
ax[1].set_ylabel('Average precision')
ax[1].legend()
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')

for a in ax:
    a.axvline(0, linestyle= '--', c='k')
    
fig.savefig('./processed_data/graph_cluster_cats/trim.png')
fig.savefig('./processed_data/graph_cluster_cats/trim.tif')
plt.close(fig)



