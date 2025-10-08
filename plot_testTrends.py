from generator import generator
from plotter import plotter
from performance import performance
from trainer import trainer

import numpy as np
import time as timeCount
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import math
import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.base import clone

from energyflow.archs import PFN

from joblib import dump, load

plt.rcParams.update({'font.size': 40,
                    #'font.family':  'Times New Roman',
                    'legend.edgecolor': 'white',
                    'xtick.minor.visible': True,
                    'ytick.minor.visible': True,
                    'xtick.major.size':15,
                    'xtick.minor.size':10,
                    'ytick.major.size':15,
                    'ytick.minor.size':10,
                    'xtick.major.width':3,
                    'xtick.minor.width':3,
                    'ytick.major.width':3,
                    'ytick.minor.width':3,
                    'axes.linewidth' : 3,
                    'figure.max_open_warning':200,
                    'lines.linewidth' : 5})


print_dir='/w/work/clas12/tyson/plots/aiDataAcceptance/toy_DR4sWeights/tests/'
endName='_100kEvents_capv2_model_2GBT'

# freqs=[1,2,4,8,10]
# asym_2to1_splot=[0.799,0.799,0.792,0.766,0.747]
# asym_2to1=[0.805,0.807,0.767,0.715,0.693]
# asymErr_2to1_splot=[0.009,0.009,0.009,0.009,0.009]
# asymErr_2to1=[0.009,0.009,0.009,0.009,0.009]

# asym_9to1_splot=[0.805,0.796,0.794,0.775,0.75]
# asym_9to1=[0.78,0.793,0.682,0.583,0.532]
# asymErr_9to1_splot=[0.027,0.027,0.027,0.027,0.027]
# asymErr_9to1=[0.027,0.026,0.0262,0.026,0.0257]

freqs=np.array([2,4,8,10])
asym_2to1_splot=[0.802,0.792,0.766,0.747]
asym_2to1=[0.807,0.767,0.715,0.693]
asymErr_2to1_splot=[0.009,0.009,0.009,0.009]
asymErr_2to1=[0.009,0.009,0.009,0.009]

asym_9to1_splot=[0.804,0.794,0.775,0.75]
asym_9to1=[.793,0.682,0.583,0.532]
asymErr_9to1_splot=[0.027,0.027,0.027,0.027]
asymErr_9to1=[0.026,0.0262,0.026,0.0257]

nGBTs=np.array([2,5,10,25])
asym_9to1_splot_nGBTs=[0.75,0.75,0.75,0.75]
asym_9to1_nGBTs=[0.532,0.59,0.639,0.741]
asymErr_9to1_splot_nGBTs=[0.027,0.027,0.027,0.027]
asymErr_9to1_nGBTs=[0.0285,0.028,0.029,0.028]
asymStd_9to1_nGBTs=[0.026,0.027,0.0336,0.0384]

fig = plt.figure(figsize = (20,20))
plt.errorbar(x=freqs+0.3, y=asym_9to1, yerr=asymErr_9to1,fmt='s', color='firebrick', label='Density Ratio',ms=15,capsize=10,elinewidth=5)
plt.errorbar(x=freqs+0.15, y=asym_9to1_splot, yerr=asymErr_9to1_splot,fmt='s', color='royalblue', label='sPlot',ms=15,capsize=10,elinewidth=5)
plt.axhline(y = 0.8, color = 'grey', linestyle = '--', label='Generated')
plt.text(5, 0.9, '1:9 Signal to Background Ration', color='black', fontsize=50, fontweight='bold', rotation=0, va='center', ha='center')
plt.ylim(0.4,1.0)
plt.xlabel('Frequency')
plt.ylabel('Mean Amplitude')
plt.title('Mean Amplitude vs Frequency')
plt.legend(loc='lower left')
plt.savefig(print_dir+'TrendAsym_freq_9to1'+endName+'.png', bbox_inches="tight")

fig = plt.figure(figsize = (20,20))
plt.errorbar(x=freqs-0.15, y=asym_2to1, yerr=asymErr_2to1,fmt='s', color='mediumorchid', label='Density Ratio',ms=15,capsize=10,elinewidth=5)
plt.errorbar(x=freqs, y=asym_2to1_splot, yerr=asymErr_2to1_splot,fmt='s', color='limegreen', label='sPlot',ms=15,capsize=10,elinewidth=5)
plt.axhline(y = 0.8, color = 'grey', linestyle = '--', label='Generated')
plt.text(5, 0.9, '1:2 Signal to Background Ration', color='black', fontsize=50, fontweight='bold', rotation=0, va='center', ha='center')
plt.ylim(0.4,1.0)
plt.xlabel('Frequency')
plt.ylabel('Mean Amplitude')
plt.title('Mean Amplitude vs Frequency')
plt.legend(loc='lower left')
plt.savefig(print_dir+'TrendAsym_freq_2to1'+endName+'.png', bbox_inches="tight")

fig = plt.figure(figsize = (20,20))
plt.errorbar(x=freqs-0.15, y=asym_2to1, yerr=asymErr_2to1,fmt='s', color='mediumorchid', label='Density Ratio (1:2 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
plt.errorbar(x=freqs, y=asym_2to1_splot, yerr=asymErr_2to1_splot,fmt='s', color='limegreen', label='sPlot (1:2 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
plt.errorbar(x=freqs+0.3, y=asym_9to1, yerr=asymErr_9to1,fmt='s', color='firebrick', label='Density Ratio (1:9 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
plt.errorbar(x=freqs+0.15, y=asym_9to1_splot, yerr=asymErr_9to1_splot,fmt='s', color='royalblue', label='sPlot (1:9 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
plt.axhline(y = 0.8, color = 'grey', linestyle = '--', label='Generated')
plt.ylim(0.4,1.1)
plt.xlabel('Frequency')
plt.ylabel('Mean Amplitude')
plt.title('Mean Amplitude vs Frequency')
plt.legend(loc='upper left')
plt.savefig(print_dir+'TrendAsym_freq'+endName+'.png', bbox_inches="tight")

fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True, sharey=False,tight_layout=True,height_ratios=[2, 1])
axs[0].errorbar(x=nGBTs, y=asym_9to1_nGBTs, yerr=asymErr_9to1_nGBTs,fmt='s', color='firebrick', label='Density Ratio (1:9 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
axs[0].errorbar(x=nGBTs, y=asym_9to1_splot_nGBTs, yerr=asymErr_9to1_splot_nGBTs,fmt='s', color='royalblue', label='sPlot (1:9 Signal to Background Ratio)',ms=15,capsize=10,elinewidth=5)
axs[0].axhline(y = 0.8, color = 'grey', linestyle = '--', label='Generated')
axs[0].legend(loc='upper left')
ymin, ymax = axs[0].get_ylim()
axs[0].set_ylim(0.4,1.1)
axs[0].set_title('Mean Amplitude vs Number of GBDTs')
axs[0].set_ylabel('Mean Amplitude')
axs[1].errorbar(x=nGBTs, y=asymErr_9to1_nGBTs,fmt='s', color='firebrick', label='Uncertainty',ms=15,capsize=10,elinewidth=5)
axs[1].errorbar(x=nGBTs, y=asymStd_9to1_nGBTs,fmt='s', color='mediumorchid', label='Standard Deviation',ms=15,capsize=10,elinewidth=5)
axs[1].legend(loc='upper left')
ymin, ymax = axs[1].get_ylim()
axs[1].set_ylim(0.02,0.05)
axs[1].set_ylabel(r'Uncertainty & $\sigma$')
plt.xlabel('Number of GBDTs')
plt.savefig(print_dir+'TrendAsym_nGBTs'+endName+'.png', bbox_inches="tight")

