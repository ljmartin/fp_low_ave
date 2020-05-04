import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm
from seaborn import kdeplot
import pymc3 as pm

##Set plotting parameters:
utils.set_mpl_params()
fp_names = utils.getNames(short=False)

nicenames =dict()
nicenames['morgan'] = 'Morgan'
nicenames['2dpharm'] = '2D\nPharm.'
nicenames['atom_pair'] = 'Atom Pair'
nicenames['erg'] = 'ErG'
nicenames['cats'] ='CATS'
nicenames['layered'] = 'Layered'
nicenames['maccs'] = 'MACCS'
nicenames['morgan_feat'] = 'Morgan\nFeat.'
nicenames['pattern'] = 'Pattern'
nicenames['rdk'] = 'RDK'
nicenames['topo_torsion'] ='Topol.\nTorsion'

fp_aps_before = dict()
fp_aps_after = dict()
for fp in fp_names:
    fp_aps_before[fp] = np.load('./processed_data/graph_fp_comparison/ap_before_'+fp+'.npy', allow_pickle=True)
    fp_aps_after[fp] = np.load('./processed_data/graph_fp_comparison/ap_after_'+fp+'.npy', allow_pickle=True)
 
aves_before_trim = np.load('processed_data/graph_fp_comparison/aves_before_trim.npy', allow_pickle=True)
aves_after_trim = np.load('processed_data/graph_fp_comparison/aves_after_trim.npy', allow_pickle=True)

targets = np.load('processed_data/graph_fp_comparison/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_fp_comparison/cutoffs.npy', allow_pickle=True)


def pmc_mean(d):
    with pm.Model() as m:
        diff = pm.Normal('diff', 0, sigma=2)
        sig = pm.HalfNormal('sig', sigma=2)
    
        likelihood = pm.Normal('y', mu=diff,
                        sigma=sig, observed=d)

        trace = pm.sample(2000, cores=2) # draw 3000 posterior samples
        return trace

pm_diffs0 = dict()
ctrl_fp = 'morgan'
control = fp_aps_before[ctrl_fp]

for fp in fp_names:
    score = fp_aps_before[fp]
    if fp==ctrl_fp:
        continue
    else:
        d = np.log10(score/control)
        trace = pmc_mean(d)
        pm_diffs0[fp]=trace


###We repeat the above exactly except for using CATS as the reference group
#(and changing the ylim):
pm_diffs1 = dict()
ctrl_fp = 'cats'
control = fp_aps_before[ctrl_fp]

for fp in fp_names:
    score = fp_aps_before[fp]
    if fp==ctrl_fp:
        continue
    else:
        d = np.log10(score/control)
        trace = pmc_mean(d)
        pm_diffs1[fp]=trace



###Plot relativ average precisions:

fig, ax = plt.subplots(nrows=2,ncols=1)
fig.set_figheight(12)
fig.set_figwidth(12)

ctrl_fp = 'morgan'
##Plot values relative to morgan:
for count, fp in enumerate(fp_names):
    if fp==ctrl_fp:
        ax[0].errorbar(count, 1, yerr=0, label=fp, fmt='o', markersize=15, mfc='white')
        ax[0].annotate(nicenames[fp], (count, 1+0.09), ha='center', va='center')
        continue
    #coarse-grained fingerprints:
    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        fm='d'
        linestyle='--'
    #high res fingerprints:
    else:
        fm = 'o'
        linestyle='-'
    tr = pm_diffs0[fp]['diff']
    hpd = pm.hpd(tr, credible_interval=0.99)
    hpd = 10**hpd
    y = 10**tr.mean()
    eb = ax[0].errorbar(count, y, yerr=np.array([y-hpd[0], hpd[1]-y])[:,None],
                      label=fp,
                      fmt='o',
                      linewidth=4,
                     markersize=15, mfc='white', capsize=3)
    
    ax[0].annotate(nicenames[fp], (count, y+hpd[1]-y+0.04), ha='center')
    eb[-1][0].set_linestyle(linestyle)

    
ctrl_fp = 'cats'
#Plot values relative to CATS:
for count, fp in enumerate(fp_names):
    if fp==ctrl_fp:
        ax[1].errorbar(count, 1, yerr=0, label=fp, fmt='o', markersize=15, mfc='white')
        ax[1].annotate(nicenames[fp], (count, 1-0.05), ha='center')
        continue
    #coarse-grained fingerprints:
    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        fm='d'
        linestyle='--'
    #high res fingerprints:
    else:
        fm = 'o'
        linestyle='-'
    tr = pm_diffs1[fp]['diff']
    hpd = pm.hpd(tr, credible_interval=0.99)
    hpd = 10**hpd
    y = 10**tr.mean()
    eb = ax[1].errorbar(count, y, yerr=np.array([y-hpd[0], hpd[1]-y])[:,None],
                      label=fp,
                      fmt='o',
                      linewidth=4,
                     markersize=15, mfc='white', capsize=3)
    
    ax[1].annotate(nicenames[fp], (count, hpd[0]-0.04), ha='center', va='center')
    eb[-1][0].set_linestyle(linestyle)
    
    
    
utils.plot_fig_label(ax[0], 'A.')
utils.plot_fig_label(ax[1], 'B.')
    
    
    
    
ax[0].axhline(1, linestyle='-', c='k', zorder=0)
ax[0].set_ylim(0.8,1.9)
ax[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax[0].set_xticklabels([])
ax[0].set_ylabel('Average precision relative to\nMorgan fingerprint')
ax[0].grid()


ax[1].axhline(1, linestyle='-', c='k', zorder=0)
ax[1].set_ylim(0.5,1.1)
ax[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax[1].set_xticklabels([])
ax[1].grid()
ax[1].set_ylabel('Average precision relative to\nCATS fingerprint')
ax[1].set_xlabel('Fingerprint', fontsize=18)

fig.savefig('./processed_data/graph_fp_comparison/fp_comparison_temp.png')
plt.close(fig)


