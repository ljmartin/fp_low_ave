import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import pymc3 as pm
import utils

from sklearn.metrics import precision_score, recall_score, roc_auc_score, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, confusion_matrix, average_precision_score, auc, precision_recall_curve

import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from tqdm import tqdm

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
sizes_before_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
sizes_after_trim = np.load('processed_data/graph_fp_comparison/sizes.npy', allow_pickle=True)
targets = np.load('processed_data/graph_fp_comparison/targets.npy', allow_pickle=True)
cutoffs = np.load('processed_data/graph_fp_comparison/cutoffs.npy', allow_pickle=True)

concat_av = np.concatenate([aves_before_trim, aves_after_trim])


for fp in fp_names:
    score = np.concatenate([fp_aps_before[fp], fp_aps_after[fp]])
    x_points = np.array(concat_av)
    #outlier mask:
    mask = score<0.9999
    x_points = x_points[mask]
    score = score[mask]
    y_points = np.log10((score)/(1-score))
    result = regress(x_points, y_points)
    with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfNormal('sigma', sigma=3)
        intercept = pm.Normal('Intercept', 0, sigma=3)
        x_coeff = pm.Normal('x', 0, sigma=3)

        # Define likelihood
        likelihood = pm.Normal('y', mu=intercept + x_coeff * concat_av,
                               sigma=sigma, observed=score)

        # Inference!
        trace = pm.sample(2000, cores=2)

        print(pm.hpd(trace['Intercept']), pm.hpd(trace['x']))


