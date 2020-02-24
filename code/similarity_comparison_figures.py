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

fig, ax = plt.subplots()

for fp in fp_names:
    scores = np.load('./processed_data/similarities/'+fp+'_roc.npy')
    low = np.load('./processed_data/similarities/'+fp+'_low.npy')
    high = np.load('./processed_data/similarities/'+fp+'_high.npy')

    ax.plot(scores, label=fp)

ax.legend()
ax.grid()
fig.savefig('./processed_data/similarities/roc.png')
