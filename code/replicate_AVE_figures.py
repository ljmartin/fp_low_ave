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


import pandas as pd
df = pd.read_csv('./processed_data/replicate_AVE/results.csv')
aves = df['ave']
ap = df['ap']
mcc = df['mcc']
ef = df['ef']
auroc = df['auroc']

fig, ax = plt.subplots(2,2)
fig.set_figheight(8)
fig.set_figwidth(12)


metrics = [auroc, ap, mcc, ef]
labels = ['A.', 'B.', 'C.', 'D.']
names = ['AUROC', 'Average precision', 'Matthews correlation\ncoefficient' ,'Enrichment factor']
lims = [[-0.05,1.05], [-0.05,1.05], [-0.05,1.05], [-0.5,21]]
for a, metric, label, name, lim in zip(ax.flatten(), metrics, labels, names, lims):
    a.scatter(aves, metric)
    a.set_ylabel(name)
    a.set_xlabel('AVE')
    a.grid()
    a.set_ylim(lim)
    utils.plot_fig_label(a, label)
    a.axvline(0, linestyle= '--', c='k')
    
fig.savefig('./processed_data/replicate_AVE/rep_AVE.png')
fig.savefig('./processed_data/replicate_AVE/rep_AVE.tif')
plt.close(fig)



