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



aves_before_trim = np.load('processed_data/replicate_AVE_cats/aves_before_trim.npy', allow_pickle=True)
aves_after_trim = np.load('processed_data/replicate_AVE_cats/aves_after_trim.npy', allow_pickle=True)
auroc_before_trim = np.load('processed_data/replicate_AVE_cats/auroc_before_trim.npy', allow_pickle=True)
auroc_after_trim = np.load('processed_data/replicate_AVE_cats/auroc_after_trim.npy', allow_pickle=True)
ap_before_trim = np.load('processed_data/replicate_AVE_cats/ap_before_trim.npy', allow_pickle=True)
ap_after_trim = np.load('processed_data/replicate_AVE_cats/ap_after_trim.npy', allow_pickle=True)

targets = np.load('processed_data/replicate_AVE_cats/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/replicate_AVE_cats/cutoffs.npy', allow_pickle=True)
sizes_before_trim = np.load('processed_data/replicate_AVE_cats/sizes_before_trim.npy', allow_pickle=True)
sizes_after_trim = np.load('processed_data/replicate_AVE_cats/sizes_after_trim.npy', allow_pickle=True)


fig, ax = plt.subplots(2,1)
fig.set_figheight(7.5)
fig.set_figwidth(5.5)
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

fig.savefig('./processed_data/replicate_AVE_cats/auroc_vs_ap.png')
fig.savefig('./processed_data/replicate_AVE_cats/auroc_vs_ap.tif')
plt.close(fig)





fig, ax = plt.subplots(2,1)
fig.set_figheight(7.5)
fig.set_figwidth(5.5)

ax[0].scatter(cutoffs, aves_before_trim, alpha=utils.ALPHA, label='Before trim')
ax[0].scatter(cutoffs, aves_after_trim, alpha=utils.ALPHA, label='After trim')
ax[0].set_xlabel('Single linkage cutoff (Dice)')
ax[0].set_ylabel('AVE')
ax[0].grid()
ax[0].legend()
utils.plot_fig_label(ax[0], 'A.')


##If plotting lines joining before to after:
for a,b,c,d in zip(aves_before_trim, aves_after_trim, ap_before_trim, ap_after_trim):
    ax[1].plot([a,b], [c,d], lw=0.2, c='k')
#continue:
ax[1].scatter(aves_before_trim, ap_before_trim, alpha=utils.ALPHA, label='Before trim')
ax[1].scatter(aves_after_trim, ap_after_trim, alpha=utils.ALPHA, label='After trim')
ax[1].legend()
ax[1].set_xlabel('AVE')
ax[1].set_ylabel('Average precision')
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')

fig.savefig('./processed_data/replicate_AVE_cats/trim.png')
fig.savefig('./processed_data/replicate_AVE_cats/trim.tif')
plt.close(fig)


#fig, ax = plt.subplots()
##This becomes a supplementary:
#ax.scatter(aves_before_trim, aves_after_trim, c=targets, alpha=utils.ALPHA)
#ax.plot([0,1],[0,1])
#ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C1', label='Target 1')
#ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C2', label='Target 2')
#ax.scatter([0],[1.5], alpha=utils.ALPHA, c='C3', label='etc...')
#ax.set_xlabel('AVE before trimming')
#ax.set_ylabel('AVE after trimming')
#ax.grid()
#ax.set_ylim(-0.15, 1.05)
#ax.legend()

#fig.savefig('./processed_data/supplementary/trim_size.png')
#fig.savefig('./processed_data/supplementary/trim_size.tif')
#plt.close(fig)





#fig, ax = plt.subplots()
#x_points = aves_before_trim.copy()
#xrange = np.linspace(np.min(x_points), np.max(x_points), 10)
#def regress(x,y):
#    X = sm.add_constant(x)
#    model = sm.OLS(y, X)
#    result = model.fit()
#    return result
#
#pe1 = (mpe.Stroke(linewidth=5, foreground='black'),
#       mpe.Stroke(foreground='white', alpha=1),
#       mpe.Normal())
#for score, label in zip([auroc_before_trim, ap_before_trim], ['AUROC', 'Average precision']):
##    fudge_factor = score<0.9999
#    fudge_factor = score<0.999999
#    score = score[fudge_factor]
#    x = x_points[fudge_factor].copy()
#    y = np.log10(score / (1-score))
#    result = regress(x,y)
#    ax.plot(xrange, result.params[0]+result.params[1]*xrange, label=label+' Slope:'+str(np.around(result.params[1],3)), path_effects=pe1)
#    ax.scatter(x,y, alpha=utils.ALPHA)
#ax.set_xlabel('AVE')
#ax.set_ylabel('Score')
#ax.legend()
#ax.grid()
#
#fig.savefig('./processed_data/supplementary/regression.png')
#fig.savefig('./processed_data/supplementary/regression.tif')
#plt.close(fig)

