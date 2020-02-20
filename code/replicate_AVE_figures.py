import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

##Set plotting parameters:
utils.set_mpl_params()


test_probas = np.load('processed_data/replicate_AVE/test_probas.npy', allow_pickle=True)
test_labels = np.load('processed_data/replicate_AVE/test_labels.npy', allow_pickle=True)
aves = np.load('processed_data/replicate_AVE/aves.npy', allow_pickle=True)
#ves = np.load('processed_data/replicate_AVE/ves.npy', allow_pickle=True)
targets = np.load('processed_data/replicate_AVE/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/replicate_AVE/cutoffs.npy', allow_pickle=True)
sizes = np.load('processed_data/replicate_AVE/sizes.npy', allow_pickle=True)


aurocs = list()
average_precisions = list()
precisions = list()
tprs = list()
fprs = list()

def calculate_weighted_AP(probas, labels, w):
    #calculate all unique thresholds required to test:
    thresholds = probas[np.argsort(probas)]
    thresholds = np.unique(thresholds)
    #broadcast thresholds columnways and perform the > operation for all probas at each threshold:
    preds = (probas > thresholds[:,np.newaxis])

    #now calculate the weighted TP, FP, FN:
    tp = (w*preds)*labels[np.newaxis]
    fp = (w*preds)*(1-labels)[np.newaxis]
    fn = (w*(1-preds))*(labels)[np.newaxis]
    precisions = tp.sum(axis=1)[:-1] / (tp.sum(axis=1)[:-1]+fp.sum(axis=1)[:-1])
    recalls = tp.sum(axis=1) / (tp.sum(axis=1) + fn.sum(axis=1))

    #now calculate weighted AP (this line is from Sklearn's average_precision_score):
    ap = -np.sum(np.diff(recalls) * np.array(precisions))
    return ap
    
    

for probas, y_test in tqdm(zip(test_probas, test_labels)):
    auroc = roc_auc_score(y_test, probas)
    average_precision = average_precision_score(y_test, probas)
    
    tn, fp, fn, tp = confusion_matrix(y_test, probas>0.5).ravel()
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    precision = precision_score(y_test, probas>0.5, zero_division=0)
    

    precisions.append(precision)
    tprs.append(tpr)
    fprs.append(fpr)
    aurocs.append(auroc)
    average_precisions.append(average_precision)


fig, ax = plt.subplots(1,3)
fig.set_figheight(4)
fig.set_figwidth(15)
plt.setp(ax, ylim=(-0.05,1.05))

ax[0].scatter(aves, aurocs, c=targets, alpha=utils.ALPHA)
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C1', label='Target 1')
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C2', label='Target 2')
ax[0].scatter([0],[1.5], alpha=utils.ALPHA, c='C3', label='etc...')
ax[0].set_xlabel('AVE score')
ax[0].set_ylabel('AUROC')
ax[0].legend()
ax[0].grid()
utils.plot_fig_label(ax[0],'A.')

ax[1].scatter(aves, average_precisions, c=targets, alpha=utils.ALPHA)
ax[1].set_xlabel('AVE score')
ax[1].set_ylabel('Average precision')
ax[1].grid()
utils.plot_fig_label(ax[1], 'B.')


ax[2].scatter(cutoffs, aves, c=targets, alpha=utils.ALPHA)
ax[2].set_xlabel('Linkage cutoff (jaccard)')
ax[2].set_ylabel('AVE score')
ax[2].grid()
utils.plot_fig_label(ax[2], 'C')


fig.savefig('./processed_data/replicate_AVE/auroc_vs_ap.png')
plt.close(fig)


fig, ax = plt.subplots(1)

ax.scatter(aves, precisions, label='Precision', c='C6', alpha=utils.ALPHA)
ax.scatter(aves, tprs, label='TPR (p>0.5)', alpha=utils.ALPHA)
ax.scatter(aves, fprs, label='FPR (p>0.5)', alpha=utils.ALPHA)
ax.set_xlabel('AVE score')
ax.set_ylabel('Score')
ax.grid()
ax.legend()

fig.savefig('./processed_data/replicate_AVE/tpr_vs_fpr.png')

#Transform to do linear regression:
#Transform to do linear regression:
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(6)

x_points = np.array(aves)
xrange = np.linspace(np.min(x_points), np.max(x_points),10)

def regress(x, y):
    X = sm.add_constant(x[~np.isinf(y_points)])
    model = sm.OLS(y_points[~np.isinf(y_points)],X)
    result = model.fit()
    return result


pe1 = (mpe.Stroke(linewidth=5, foreground='black'),
       mpe.Stroke(foreground='white',alpha=1),
       mpe.Normal())


for score, label in zip([np.array(average_precisions), np.array(aurocs)], ['AP', 'AUROC']):
    x_points = np.array(aves)
    x_points = x_points[score!=1]

    score = score[score!=1]
    print(score)
    y_points = np.log10((score)/(1-score))
    
    result = regress(x_points, y_points)
    ax.plot(xrange, result.params[0]+result.params[1]*xrange, label=label+' $R^2$: '+str(np.around(result.rsquared,3)), path_effects=pe1)    
    ax.scatter(x_points, y_points, alpha=utils.ALPHA)
ax.set_xlabel('AVEs')
ax.set_ylabel('Score')
ax.legend()

##For repeating the above using VE score
#x_points = np.array(ves)
#for score, label in zip([np.array(average_precisions), np.array(aurocs)], ['AP', 'AUROC']):
#    y_points = np.log10((score)/(1-score))    
#    result = regress(x_points, y_points)
#    ax[1].plot(xrange, result.params[0]+result.params[1]*xrange, label=label+' R^2: '+str(np.around(result.rsquared,3)))    
#    ax[1].scatter(x_points, y_points, s=10)
#ax[1].set_xlabel('VEs')
#ax[1].set_ylabel('Score')
#ax[1].legend()


fig.savefig('./processed_data/replicate_AVE/regression.png')
plt.close(fig)

