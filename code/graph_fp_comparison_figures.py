import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import pandas as pd

import utils
from scipy.special import logit 
from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm
from seaborn import kdeplot
import pymc3 as pm

##Set plotting parameters:
utils.set_mpl_params()

fname = './processed_data/graph_cluster_both/df_before_trim.csv'
before_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_cluster_both/df_after_morgan_trim.csv'
after_morgan_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_cluster_both/df_after_cats_trim.csv'
after_cats_trim = pd.read_csv(fname, index_col=0)



fig, ax = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(8)
kdeplot(after_morgan_trim['ave_morgan'], ax=ax[0], label='$AVE_{Morgan}$', c='C0')
kdeplot(after_cats_trim['ave_cats'], ax=ax[0], label='$AVE_{CATS}$', c='C1', )

num = len(after_morgan_trim['ave_morgan'])
ax[0].scatter(after_morgan_trim['ave_morgan'], np.random.uniform(-1, -0.5, num), c='C0', alpha=0.05)
ax[0].scatter(after_cats_trim['ave_cats'], np.random.uniform(-0.5, 0.0, num), c='C1',alpha=0.05)

ax[0].set_title('AVE achieved after debiasing')
ax[0].set_xlabel('AVE')
ax[0].set_ylabel('Density')
ax[0].grid()
ax[0].legend()
ax[0].axvline(0, linestyle='--', c='k')

utils.plot_fig_label(ax[0], 'A.')





####Now do some arithmetic-mean estimation:
def calc_hpd(data, statistic=np.mean):
    with pm.Model() as model:
        #prior on statistic of interest:
        a = pm.Normal('a', mu=statistic(data), sigma=10.0)
        #'nuisance' parameter:
        b = pm.HalfNormal('b', sigma=10.0)
        
        #likelihood:
        if statistic==np.mean:
            y = pm.Normal('y', mu = a, sigma = b, observed=data)
        elif statistic==np.median:
            y = pm.Laplace('y', mu = a, b = b, observed=data)
        trace = pm.sample(draws=1000, tune=500, chains=2, target_accept=0.9)
    return trace


cutoffs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15]

MAPs = list()
HPD_high = list()
HPD_low = list()


for cutoff in cutoffs:
    #generate masks that ensure all AVE's are within the +/- the cutoff:
    mask = (after_morgan_trim['ave_morgan']<cutoff) & (after_cats_trim['ave_cats']<cutoff)
    mask = np.logical_and(mask, ((after_morgan_trim['ave_morgan']>-cutoff) & (after_cats_trim['ave_cats']>-cutoff)))
    
    ##logit:
    #trace = calc_hpd(logit(after_cats_trim['ap_cats'][mask]) - logit(after_morgan_trim['ap_morgan'][mask]))
    #max_a_posteriori = np.mean(trace['a'])
    #hpd = pm.stats.hpd(trace['a'], credible_interval=0.95)

    #relative:
    trace = calc_hpd( np.log(after_cats_trim['ap_cats'][mask] / after_morgan_trim['ap_morgan']) )
    max_a_posteriori = np.exp(np.mean(trace['a']))
    hpd = np.exp( pm.stats.hpd(trace['a'], credible_interval=0.95) )

    MAPs.append(max_a_posteriori)
    HPD_high.append(hpd[1])
    HPD_low.append(hpd[0])

ax[1].axhline(1,c='k', linestyle='--')

ax[1].set_title('Relative performance of CATS and\nMorgan fingerprints after debiasing')
ax[1].fill_between(x=cutoffs, y2=HPD_high, y1=HPD_low, label='95% credible region')
#ax[1].plot(cutoffs, HPD_high, 'k', marker='o', mfc='white')
#ax[1].plot(cutoffs, HPD_low, 'k', marker='o', mfc='white')
ax[1].plot(cutoffs, HPD_high, 'k', )
ax[1].plot(cutoffs, HPD_low, 'k', )
ax[1].plot(cutoffs, MAPs, marker='o', color='k',mfc='white', mec='k', label='Max. a posteriori estimate')

ax[1].legend()
ax[1].set_xlabel('AVE cutoff, +/-')
ax[1].set_ylabel('Average precision relative\nto Morgan fingerprint')



fig.savefig('./processed_data/graph_fp_comparison/ave_distribution.png')
fig.savefig('./processed_data/graph_fp_comparison/ave_distribution.tif')
plt.close(fig)
