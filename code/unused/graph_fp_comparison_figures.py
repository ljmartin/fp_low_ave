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
#sizes_before_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
#sizes_after_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
targets = np.load('processed_data/graph_fp_comparison/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_fp_comparison/cutoffs.npy', allow_pickle=True)






fig, ax = plt.subplots()
kdeplot(aves_before_trim,ax=ax, label='Before trim')
plt.scatter(aves_before_trim,
            np.zeros(aves_before_trim.shape[0])+np.random.uniform(-0.5, 0.5,aves_before_trim.shape[0]),
            alpha=0.2)
kdeplot(aves_after_trim, ax=ax, label='After trim')
ax.scatter(aves_after_trim,
            np.zeros(aves_after_trim.shape[0])+np.random.uniform(-0.5, 0.5,aves_after_trim.shape[0]),
           alpha=0.2)
ax.set_xlabel('AVE')
ax.set_ylabel('Density')
ax.grid()
ax.legend()
fig.savefig('./processed_data/graph_fp_comparison/ave_distribution.png')
fig.savefig('./processed_data/graph_fp_comparison/ave_distribution.tif')
plt.close(fig)



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
control = fp_aps_after[ctrl_fp]

for fp in fp_names:
    score = fp_aps_after[fp]
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
control = fp_aps_after[ctrl_fp]

for fp in fp_names:
    score = fp_aps_after[fp]
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

fig.savefig('./processed_data/graph_fp_comparison/fp_comparison.png')
fig.savefig('./processed_data/graph_fp_comparison/fp_comparison.tif')
plt.close(fig)








####Now we regress on AP vs AVE:
def do_bayes_regression(x,y):
    with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfNormal('sigma', sigma=3)
        intercept = pm.Normal('Intercept', 0, sigma=3)
        x_coeff = pm.Normal('x', 0, sigma=3)

        # Define likelihood
        likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                        sigma=sigma, observed=y)

        # Inference!
        trace = pm.sample(2000, cores=2) # draw 3000 posterior samples using NUTS 
        return trace



pm_traces = dict()

#here we regress on the merged average precisions, i.e. before trimming AND after trimming,
#to get more data.
concat_ave = np.concatenate([aves_before_trim, aves_after_trim])

for fp in fp_names:
    score = np.concatenate([fp_aps_before[fp], fp_aps_after[fp]])
    #    score = fp_aps_after[fp]
    #    x_points = np.array(aves_after_trim)
    x_points = np.array(concat_ave)
    y_points = np.log10((score)/(1-score))
    trace = do_bayes_regression(x_points, y_points)
    pm_traces[fp]=trace




##Plot a three tier figure showing regressions:
fig, ax = plt.subplots(3, 1)
fig.set_figheight(10)
fig.set_figwidth(10)

labels=list()
for count, fp in enumerate(fp_names):
    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        fm='d'
        linestyle='--'                                                                                                                                                             
    else:
        fm = 'o'
        linestyle='-'
    tr = pm_traces[fp]['Intercept']
    hpd = pm.hpd(tr,  credible_interval=0.99)
    y = tr.mean()
    eb = ax[0].errorbar(count, tr.mean(), yerr=np.array([hpd[0]-y, y-hpd[1]])[:,None], label=fp, fmt='o',
                  linewidth=4, markersize=15, mfc='white', capsize=3)
    eb[-1][0].set_linestyle(linestyle)
    labels.append(nicenames[fp])

ax[0].set_ylabel('Intercept\n(higher is better)')
ax[0].grid()
ax[0].set_xticklabels([])
ax[0].set_xticks(np.arange(count+1))
ax[0].tick_params(direction='out', pad=25)
ax[0].set_xticklabels(labels, va='center')
utils.plot_fig_label(ax[0], 'A.')

for count, fp in enumerate(fp_names):
    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        fm='d'
        linestyle='--'                                                                                                                                                             
    else:
        fm = 'o'
        linestyle='-'
    tr = pm_traces[fp]['x']
    hpd = pm.hpd(tr,  credible_interval=0.99)
    y = tr.mean()
    eb = ax[1].errorbar(count, tr.mean(), yerr=np.array([hpd[0]-y, y-hpd[1]])[:,None], label=fp, fmt='o',
                  linewidth=4, markersize=15, mfc='white', capsize=3)
    eb[-1][0].set_linestyle(linestyle)
    
ax[1].set_ylabel('Slope\n(lower is better)')
ax[1].grid()
ax[1].tick_params(bottom=False, top=True, left=True, right=False)
ax[1].set_xticklabels([])
ax[1].set_xticks(np.arange(count+1))
utils.plot_fig_label(ax[1], 'B.')

def transform_y(y):
    y = 10**y
    return y / (1+y)

xr = np.linspace(-0.2,0.5,100)

for count, color, name in zip([0, 1], [0,4],['morgan', 'cats']):
    intercept = pm_traces[name]['Intercept'].mean()
    xcoef = pm_traces[name]['x'].mean()
    y = xr*xcoef+intercept
    y = transform_y(y)
    ax[2].plot(xr, y, lw=2, label=name, c='C'+str(color))

    ##This is showing the boundaries of epsilon (called sigma here for some reason):
    y2 = transform_y(xr*xcoef+intercept-pm_traces[name]['sigma'].mean()*1.96)
    y1 = transform_y(xr*xcoef+intercept+pm_traces[name]['sigma'].mean()*1.96)


    ax[2].fill_between(xr, y1, y2, alpha=0.3, color='C'+str(color))
    ax[2].legend()
    ax[2].grid()
    ax[2].axvline(0, linestyle='--', c='k', lw=1)

ax[2].set_ylabel('Average precision')
ax[2].set_xlabel('AVE')
utils.plot_fig_label(ax[2], 'C.')

fig.savefig('./processed_data/graph_fp_comparison/regression_comparison.png')
fig.savefig('./processed_data/graph_fp_comparison/regression_comparison.tif')
plt.close(fig)



####Plotting select regressions to supplementary:
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(12)
for count, name in enumerate(fp_names):
#    if name in ['morgan', 'morgan_feat', 'cats', 'atom_pair', 'erg']:
#        pass
#    else:
#        continue
    intercept = pm_traces[name]['Intercept'].mean()
    xcoef = pm_traces[name]['x'].mean()
    y = xr*xcoef+intercept
    y = transform_y(y)
    ax.plot(xr, y, lw=2, c='C'+str(count), label=nicenames[name])

    ##This is showing the boundaries of epsilon (called sigma here for some reason):
    #y2 = transform_y(xr*xcoef+intercept-pm_traces[name]['sigma'].mean()*1.96)
    #y1 = transform_y(xr*xcoef+intercept+pm_traces[name]['sigma'].mean()*1.96)

#    ax[count].fill_between(xr, y1, y2, alpha=0.3, color='C'+str(count))

ax.axvline(0, linestyle='--', c='k', lw=1)
ax.grid()
ax.legend()
fig.savefig('./processed_data/supplementary/fp_regr_comparison.png')
fig.savefig('./processed_data/supplementary/fp_regr_comparison.tif')
plt.close(fig)



#
#fig, ax = plt.subplots(len(fp_names), 1)
#fig.set_figheight(20)
#fig.set_figwidth(10)
#for count, name in enumerate(fp_names):
#
#    intercept = pm_traces[name]['Intercept'].mean()
#    xcoef = pm_traces[name]['x'].mean()
#    y = xr*xcoef+intercept
#    y = transform_y(y)
#    ax[count].plot(xr, y, lw=2, label=name, c='C'+str(count))
#
#    ##This is showing the boundaries of epsilon (called sigma here for some reason):
#    y2 = transform_y(xr*xcoef+intercept-pm_traces[name]['sigma'].mean()*1.96)
#    y1 = transform_y(xr*xcoef+intercept+pm_traces[name]['sigma'].mean()*1.96)
#
#    ax[count].fill_between(xr, y1, y2, alpha=0.3, color='C'+str(count))
#    ax[count].legend()
#    ax[count].grid()
#    ax[count].axvline(0, linestyle='--', c='k', lw=1)
#
#fig.savefig('./processed_data/supplementary/all_regr_comparison.png')
#plt.close(fig)


