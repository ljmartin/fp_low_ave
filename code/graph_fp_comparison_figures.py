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

##load up files:
fname = './processed_data/graph_fp_comparison/df_before_trim.csv'
before_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_fp_comparison/df_after_morgan_trim.csv'
after_morgan_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_fp_comparison/df_after_cats_trim.csv'
after_cats_trim = pd.read_csv(fname, index_col=0)


##This function calculates the mean (or median if desired) of
##the input data using MCMC (actually No-U-Turn-Sampling thru PyMC)
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


##get the MCMC samples of log(cats_ap  /  morgan_ap)
tr = calc_hpd(np.log(after_cats_trim['ap_cats'] / after_morgan_trim['ap_morgan']))



fig, ax = plt.subplots(2,2)
fig.set_figwidth(13)
fig.set_figheight(10)


hpd = np.exp(pm.hpd(tr['a']))
#hpd = pm.hpd(tr['a'])
#plot AVE scores KDEs:    
kdeplot(after_morgan_trim['ave_morgan'], ax=ax[0,0], label='AVE$_{Morgan}$')
kdeplot(after_cats_trim['ave_cats'], ax=ax[0,0], label='AVE$_{CATS}$')
ax[0,0].set_ylabel('Density')
ax[0,0].set_title('AVE after debiasing')
utils.plot_fig_label(ax[0,0], 'A.')



#plot AVE vs AP:
ax[0,1].scatter(after_morgan_trim['ave_morgan'], after_morgan_trim['ap_morgan'], 
                c='C0', label='Morgan', zorder=0, alpha=0.85)
ax[0,1].scatter(after_cats_trim['ave_cats'], after_cats_trim['ap_cats'], 
                c='C1', label='CATS',zorder=1, alpha=0.65)
ax[0,1].axvline(0, linestyle='--', c='k')
ax[0,1].set_ylabel('Average Precision (AP)')
ax[0,1].legend()
ax[0,1].set_title('Bias - performance relationship')
utils.plot_fig_label(ax[0,1], 'B.')


#plot one-tailed estimate:
Z = np.exp(tr['a'])
#Z = tr['a']
N = len(Z)
H,X1 = np.histogram( Z, bins = 3000, density=True)
dx = X1[1] - X1[0]
F1 = np.cumsum(H)*dx
ax[1,0].plot(X1[1:], 1-F1, c='C5')
ax[1,0].axvline(1, c='k', linestyle='--',)
ax[1,0].set_ylabel('Probability')
ax[1,0].set_xlabel('$\dfrac{AP_{CATS}}{AP_{Morgan}}$')
ax[1,0].set_title('Probability of CATS vs Morgan performance')
utils.plot_fig_label(ax[1,0], 'C.')

#plot two-tailed estimate:
kdeplot(np.exp(tr['a']), ax=ax[1,1], c='C5')
#kdeplot(tr['a'], ax=ax[1,1], c='C5')
ax[1,1].plot([hpd[0],hpd[1]],[0,0],'-o', c='C5', label='95% credible region')
#ax[1,1].scatter(tr['a'].mean(), 0, s= 300, facecolor='white',zorder=10,edgecolor='C5')
ax[1,1].scatter(np.exp(tr['a'].mean()), 0, s= 300, facecolor='white',zorder=10,edgecolor='C5')
ax[1,1].set_xlabel('$\dfrac{AP_{CATS}}{AP_{Morgan}}$')
ax[1,1].set_ylabel('Density')
ax[1,1].set_title('Estimated CATS vs Morgan performance')
ax[1,1].legend()
utils.plot_fig_label(ax[1,1], 'D.')

for a in ax.flatten():
    a.grid()
#     #a.axvline(l, linestyle='--', c='k')
#     #a.legend()

    
ax[1,0].axvline(1, linestyle='--', c='k')
ax[1,1].axvline(1, linestyle='--', c='k')
ax[0,0].axvline(0, linestyle='--', c='k')
ax[0,0].set_xlabel('AVE')
ax[0,1].set_xlabel('AVE')


fig.savefig('./processed_data/graph_fp_comparison/comparison.png')
