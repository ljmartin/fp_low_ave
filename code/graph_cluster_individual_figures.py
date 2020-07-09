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

c_aves_before_trim = np.load('./processed_data/graph_cluster_cats/aves_before_trim.npy')
c_aves_after_trim = np.load('./processed_data/graph_cluster_cats/aves_after_trim.npy')
c_ap_before_trim = np.load('./processed_data/graph_cluster_cats/ap_before_trim.npy')
c_ap_after_trim = np.load('./processed_data/graph_cluster_cats/ap_after_trim.npy')

m_aves_before_trim = np.load('./processed_data/graph_cluster/aves_before_trim.npy')
m_aves_after_trim = np.load('./processed_data/graph_cluster/aves_after_trim.npy')
m_ap_before_trim = np.load('./processed_data/graph_cluster/ap_before_trim.npy')
m_ap_after_trim = np.load('./processed_data/graph_cluster/ap_after_trim.npy')

targets = np.load('processed_data/graph_cluster_cats/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_cluster_cats/cutoffs.npy', allow_pickle=True)
#sizes = np.load('processed_data/graph_cluster/sizes.npy', allow_pickle=True)


fig, ax = plt.subplots(2,2)
fig.set_figheight(7.5)
fig.set_figwidth(10.0)

morgan = [m_aves_before_trim, m_aves_after_trim, m_ap_before_trim, m_ap_after_trim]
cats = [c_aves_before_trim, c_aves_after_trim, c_ap_before_trim, c_ap_after_trim]

for column, results, title in zip([0,1], [morgan, cats], ['Morgan fingerprint', 'CATS fingerprint']):
    
    mu_before, sigma_before = norm.fit(results[0])
    mu_after, sigma_after = norm.fit(results[1])

    kdeplot(results[0], ax=ax[0,column], label=f'Before trim, \nμ={np.around(mu_before,3)}, σ={np.around(sigma_before,3)}')
    kdeplot(results[1], ax=ax[0,column], label=f'After trim, \nμ={np.around(mu_after, 3)}, σ={np.around(sigma_after,3)}')
    ax[0,column].set_ylabel('Density')
    ax[0,column].set_xlabel('AVE')
    ax[0,column].grid()
    ax[0,column].legend()
    ax[0,column].axvline(0, linestyle='--', c='k')
    ax[0, column].set_title(title)


    for a,b,c,d in zip(results[0], results[1], results[2], results[3]):
        ax[1,column].plot([a,b], [c,d], lw=0.2, c='k')
    ax[1,column].scatter(results[0], results[2], alpha=utils.ALPHA, label='Before trim')
    ax[1,column].scatter(results[1], results[3], alpha=utils.ALPHA, label='After trim')
    ax[1,column].set_xlabel('AVE')
    ax[1,column].set_ylabel('Average precision')
    ax[1,column].legend()
    ax[1,column].grid()


utils.plot_fig_label(ax[0,0], 'A.')
utils.plot_fig_label(ax[1,0], 'B.')
utils.plot_fig_label(ax[0,1], 'C.')
utils.plot_fig_label(ax[1,1], 'D.')
fig.savefig('./processed_data/graph_cluster/trim_individual.png')
fig.savefig('./processed_data/graph_cluster/trim_individual.tif')
plt.close(fig)



