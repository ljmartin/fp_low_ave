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


fp_aps_before = dict()
fp_aps_after = dict()
for fp in fp_names:
    fp_aps_before[fp] = np.load('./processed_data/graph_fp_comparison/ap_before_'+fp+'.npy', allow_pickle=True)
    fp_aps_after[fp] = np.load('./processed_data/graph_fp_comparison/ap_after_'+fp+'.npy', allow_pickle=True)
 
aves_before_trim = np.load('processed_data/graph_fp_comparison/aves_before_trim.npy', allow_pickle=True)
aves_after_trim = np.load('processed_data/graph_fp_comparison/aves_after_trim.npy', allow_pickle=True)
#sizes_before_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
#sizes_after_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
targets = np.load('processed_data/graph_fp_comparison/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_fp_comparison/cutoffs.npy', allow_pickle=True)






fig, ax = plt.subplots()
kdeplot(aves_before_trim,ax=ax)
plt.scatter(aves_before_trim,
            np.zeros(aves_before_trim.shape[0])+np.random.uniform(-0.5, 0.5,aves_before_trim.shape[0]),
            alpha=0.2,
            label='Before trim')
kdeplot(aves_after_trim, ax=ax)
ax.scatter(aves_after_trim,
            np.zeros(aves_after_trim.shape[0])+np.random.uniform(-0.5, 0.5,aves_after_trim.shape[0]),
           alpha=0.2,
           label='After trim')
ax.set_xlabel('AVE')
ax.set_ylabel('Density')
ax.grid()
fig.savefig('./processed_data/graph_fp_comparison/ave_distribution.png')







def pmc_mean(d):
    with pm.Model() as m:
        diff = pm.Normal('diff', 0, sigma=2)
        sig = pm.HalfNormal('sig', sigma=2)
    
        likelihood = pm.Normal('y', mu=diff,
                        sigma=sig, observed=d)

        trace = pm.sample(2000, cores=2) # draw 3000 posterior samples
        return trace



pm_diffs = dict()
ctrl_fp = 'morgan'
control = fp_aps_after[ctrl_fp]

for fp in fp_names:
    score = fp_aps_after[fp]
    if fp==ctrl_fp:
        continue
    else:
        d = np.log10(score/control)
        trace = pmc_mean(d)
        pm_diffs[fp]=trace


fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(12)
for count, fp in enumerate(fp_names):
    if fp==ctrl_fp:
        plt.errorbar(count, 1, yerr=0, label=fp, fmt='o', markersize=15, mfc='white')
        continue
    #coarse-grained fingerprints:
    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        fm='d'
        linestyle='--'
    #high res fingerprints:
    else:
        fm = 'o'
        linestyle='-'
    tr = pm_diffs[fp]['diff']
    hpd = pm.hpd(tr)
    hpd = 10**hpd
    y = 10**tr.mean()
    eb = plt.errorbar(count, y, yerr=np.array([y-hpd[0], hpd[1]-y])[:,None], 
                      label=fp, 
                      fmt='o',
                      linewidth=4,
                     markersize=15, mfc='white', capsize=3)
    eb[-1][0].set_linestyle(linestyle)

ax.axhline(1, linestyle='-', c='k', zorder=0)
ax.legend(ncol=4, loc=3)
ax.set_ylim(0.6,1.8)
ax.set_ylabel('Relative average-precision')
ax.set_xlabel('Fingerprint')
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10], [])
ax.grid()

fig.savefig('./processed_data/graph_fp_comparison/fp_comparison.png')



#####
###For regression of AVE vs. average precision. 
#####


#concat_av = np.concatenate([aves_before_trim, aves_after_trim])
#fig, ax = plt.subplots(1)
#for fp in fp_names:
#    concat_ap = np.concatenate([fp_aps_before[fp], fp_aps_after[fp]])
#    ax.scatter(concat_av+np.random.uniform(-0.01,0.01, len(concat_av)), concat_ap, s=25, alpha=utils.ALPHA, label=fp)
#    
#ax.set_xlabel('AVE score')
#ax.set_ylabel('AP')
#ax.grid()    
#ax.legend()
#fig.savefig('./processed_data/graph_fp_comparison/ap_all.png')
#plt.close(fig)



#fig, ax = plt.subplots()
#fig.set_figwidth(10)
#fig.set_figheight(8)
#xrange = np.linspace(np.min(concat_av), np.max(concat_av),10)
#def regress(x, y):
#    X = sm.add_constant(x[~np.isinf(y_points)])
#    model = sm.OLS(y_points[~np.isinf(y_points)],X)
#    result = model.fit()
#    return result
#
#pe1 = (mpe.Stroke(linewidth=1, foreground='black'),
#       mpe.Stroke(foreground='white',alpha=1),
#       mpe.Normal())
#
#for fp in fp_names:
#    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
#        linestyle='--'
#    else:
#        linestyle='-'
#    score = np.concatenate([fp_aps_before[fp], fp_aps_after[fp]])
#    x_points = np.array(concat_av)
#    #outlier mask:
#    mask = score<0.99999
#    x_points = x_points[mask]
#    score = score[mask]
#    y_points = np.log10((score)/(1-score))
#    result = regress(x_points, y_points)
#    ax.plot(xrange, result.params[0]+result.params[1]*xrange,
#            label=fp+' $R^2$: '+str(np.around(result.rsquared,3)),
#            path_effects=pe1,
#            alpha=0.5, linewidth=3, linestyle=linestyle)
#    ax.scatter(x_points+np.random.uniform(-0.01, 0.01, len(x_points)), y_points, s=25, linewidth=0.4, alpha=utils.ALPHA)
#    print(fp, result.params[0], result.params[1])
#
#ax.set_xlabel('AVEs')
#ax.set_ylabel('Score')
#ax.legend(loc=9, ncol=2)
#fig.savefig('./processed_data/graph_fp_comparison/regression.png')
#plt.close(fig)
