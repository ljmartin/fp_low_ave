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
fp_names = utils.getNames()

fp_probas = dict()
fp_average_precisions = dict()
for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue
    
    fp_probas[fp] = np.load('./processed_data/fp_comparison/'+fp+'_probas.npy', allow_pickle=True)
    fp_average_precisions[fp] = []

test_labels = np.load('processed_data/fp_comparison/test_labels.npy', allow_pickle=True)
aves = np.load('processed_data/fp_comparison/aves.npy', allow_pickle=True)
targets = np.load('processed_data/fp_comparison/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/fp_comparison/cutoffs.npy', allow_pickle=True)
sizes = np.load('processed_data/fp_comparison/sizes.npy', allow_pickle=True)


average_precisions = list()

for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue

    test_probas = fp_probas[fp]
    for probas, y_test in tqdm(zip(test_probas, test_labels)):
        average_precision = average_precision_score(y_test, probas)    
        fp_average_precisions[fp].append(average_precision)


for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue

    fig, ax = plt.subplots(1)
    ax.scatter(aves, fp_average_precisions[fp], c=targets, alpha=utils.ALPHA)
    ax.set_xlabel('AVE score')
    ax.set_ylabel('AP')
    ax.grid()    
    fig.savefig('./processed_data/fp_comparison/ap_'+fp+'.png')
    plt.close(fig)

fig, ax = plt.subplots(1)
for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue

    ax.scatter(np.array(aves)+np.random.uniform(-0.01,0.01, len(aves)), fp_average_precisions[fp], s=25, alpha=utils.ALPHA, label=fp)

    
ax.set_xlabel('AVE score')
ax.set_ylabel('AP')
ax.grid()    
ax.legend()
fig.savefig('./processed_data/fp_comparison/ap_all.png')
plt.close(fig)



fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(8)
xrange = np.linspace(np.min(aves), np.max(aves),10)
def regress(x, y):
    X = sm.add_constant(x[~np.isinf(y_points)])
    model = sm.OLS(y_points[~np.isinf(y_points)],X)
    result = model.fit()
    return result

pe1 = (mpe.Stroke(linewidth=1, foreground='black'),
       mpe.Stroke(foreground='white',alpha=1),
       mpe.Normal())

for fp in fp_names:
    if fp not in ['morgan', 'cats']:
        continue

    if fp in ['cats', 'erg', '2dpharm', 'morgan_feat', 'maccs']:
        linestyle='--'
    else:
        linestyle='-'
    score = np.array(fp_average_precisions[fp])
    x_points = np.array(aves)
    #outlier mask:
    mask = score<0.99999
    x_points = x_points[mask]
    score = score[mask]
    y_points = np.log10((score)/(1-score))
    result = regress(x_points, y_points)
    ax.plot(xrange, result.params[0]+result.params[1]*xrange,
            label=fp+' $R^2$: '+str(np.around(result.rsquared,3)),
            path_effects=pe1,
            alpha=0.5, linewidth=3, linestyle=linestyle)
    ax.scatter(x_points+np.random.uniform(-0.01, 0.01, len(x_points)), y_points, s=25, linewidth=0.4, alpha=utils.ALPHA)
    print(fp, result.params[0])

ax.set_xlabel('AVEs')
ax.set_ylabel('Score')
ax.legend(loc=9, ncol=2)
fig.savefig('./processed_data/fp_comparison/regression.png')
plt.close(fig)
