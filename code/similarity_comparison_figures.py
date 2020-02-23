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

