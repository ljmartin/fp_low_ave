import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from seaborn import kdeplot

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

##Set plotting parameters:
utils.set_mpl_params()



aves_before_trim = np.load('processed_data/replicate_AVE/aves_before_trim.npy', allow_pickle=True)
aves_after_trim = np.load('processed_data/replicate_AVE/aves_after_trim.npy', allow_pickle=True)
auroc_before_trim = np.load('processed_data/replicate_AVE/auroc_before_trim.npy', allow_pickle=True)
auroc_after_trim = np.load('processed_data/replicate_AVE/auroc_after_trim.npy', allow_pickle=True)
ap_before_trim = np.load('processed_data/replicate_AVE/ap_before_trim.npy', allow_pickle=True)
ap_after_trim = np.load('processed_data/replicate_AVE/ap_after_trim.npy', allow_pickle=True)

targets = np.load('processed_data/replicate_AVE/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/replicate_AVE/cutoffs.npy', allow_pickle=True)
sizes_before_trim = np.load('processed_data/replicate_AVE/sizes_before_trim.npy', allow_pickle=True)
sizes_after_trim = np.load('processed_data/replicate_AVE/sizes_after_trim.npy', allow_pickle=True)


fig, ax = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)
plt.setp(ax, ylim=(-0.05,1.05))

ax[0].scatter(aves_before_trim, auroc_before_trim, c=targets, alpha=utils.ALPHA)
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C1', label='Target 1')
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C2', label='Target 2')
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C3', label='etc...')
ax[0].set_xlabel('AVE score')
ax[0].set_ylabel('AUROC')
ax[0].legend()
ax[0].grid()
utils.plot_fig_label(ax[0],'A.')

ax[1].scatter(aves_before_trim, ap_before_trim, c=targets, alpha=utils.ALPHA)
ax[1].set_xlabel('AVE score')
ax[1].set_ylabel('Average precision')
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')

fig.savefig('./processed_data/replicate_AVE/auroc_vs_ap.png')
plt.close(fig)





fig, ax = plt.subplots(1,3)
fig.set_figheight(6)
fig.set_figwidth(18)

ax[0].scatter(cutoffs, aves_before_trim, alpha=utils.ALPHA, label='Before trim')
ax[0].scatter(cutoffs, aves_after_trim, alpha=utils.ALPHA, label='After trim')
ax[0].set_xlabel('Single linkage cutoff (Dice)')
ax[0].set_ylabel('AVE')
ax[0].grid()
ax[0].legend()
utils.plot_fig_label(ax[0], 'A.')

ax[1].scatter(aves_before_trim, aves_after_trim, c=targets, alpha=utils.ALPHA)
ax[1].plot([0,1],[0,1])
ax[1].set_xlabel('AVE before trimming')
ax[1].set_ylabel('AVE after trimming')
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')

ax[2].scatter(aves_before_trim, ap_before_trim, alpha=utils.ALPHA, label='Before trim')
ax[2].scatter(aves_after_trim, ap_after_trim, alpha=utils.ALPHA, label='After trim')
ax[2].legend()
ax[2].set_xlabel('AVE')
ax[2].set_ylabel('AP')
ax[2].grid()
utils.plot_fig_label(ax[2], 'C.')


fig.savefig('./processed_data/replicate_AVE/trim.png')


fig, ax = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)


ax[0].scatter(aves_before_trim, sizes_before_trim[:,[0,1]].sum(1) / sizes_before_trim[:,[2,3]].sum(1),
           alpha=utils.ALPHA,
           label='Before trim')
ax[0].scatter(aves_after_trim, sizes_after_trim[:,[0,1]].sum(1) / sizes_after_trim[:,[2,3]].sum(1),
           alpha=utils.ALPHA,
           label='After trim')
ax[0].set_xlabel('AVE')
ax[0].set_ylabel('Test size / Train size')
ax[0].grid()
ax[0].legend()
utils.plot_fig_label(ax[0], 'A.')

kdeplot( np.log10(sizes_before_trim[:,[0,1]].sum(1) / sizes_before_trim[:,[2,3]].sum(1)), ax=ax[1], label='Before trim')
kdeplot( np.log10(sizes_after_trim[:,[0,1]].sum(1) / sizes_after_trim[:,[2,3]].sum(1)), ax=ax[1], label='After trim')
ax[1].set_xlabel('Log$_10$(Test size / Train size)')
ax[1].set_ylabel('Density')
utils.plot_fig_label(ax[1], 'B')


fig.savefig('./processed_data/replicate_AVE/sizes.png')
