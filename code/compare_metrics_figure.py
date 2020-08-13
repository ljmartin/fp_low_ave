import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import pandas as pd
import statsmodels.api as sm
import utils


utils.set_mpl_params()

fname = './processed_data/graph_fp_comparison/df_before_trim.csv'
before_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_fp_comparison/df_after_morgan_trim.csv'
after_morgan_trim = pd.read_csv(fname, index_col=0)

fname = './processed_data/graph_fp_comparison/df_after_cats_trim.csv'
after_cats_trim = pd.read_csv(fname, index_col=0)


fig, ax = plt.subplots(1,3)
fig.set_figwidth(15)

count = 0
for metric, title in zip(['ap', 'mcc', 'ef'], ['Average Precision', 'Matthews correlation coefficient','Enrichment factor']):
    cats = after_cats_trim[metric+'_cats']
    morg = after_morgan_trim[metric+'_morgan']
    
    for data, lab in zip([cats, morg], ['CATS','Morgan']):
        dens = sm.nonparametric.KDEUnivariate(data)
        dens.fit()
        x =np.linspace(0,max(1, max(max(cats), max(morg))),100)
        y = dens.evaluate(x)
        ax[count].plot(x,y, label=lab)
    ax[count].legend()
    ax[count].set_title(title)
    ax[count].set_ylabel('Density')
    ax[count].set_xlabel('Metric')
    ax[count].set_yticklabels([])
    #ax[count].set_yticks([])
    ax[count].grid()
    count+=1


    
utils.plot_fig_label(ax[0], 'A.')
utils.plot_fig_label(ax[1], 'B.')
utils.plot_fig_label(ax[2], 'C.')

fig.savefig('./processed_data/graph_fp_comparison/metrics.png')
