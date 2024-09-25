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

#plots outputted to print_dir
#append endName to end of plots to avoid overwriting plots
print_dir=''
endName='_1000kEvents_9to1_BGtoSig_GBDTandGBDT' 

print('\nPlots will be written to directory: '+print_dir)
print('Plot names formatted as name'+endName+'.pdf\n\n')

#change plotting style
plotWithErrorBars=True

#change number of generated events
nEvents=1000000
#nEvents=3334

#change signal to background ratio
BGtoSigRatio=(9,1)

#variable ranges
mRange=(0,10)
phiRange=(-np.pi,np.pi)
ZRange=(-1,1)

pter=plotter(mRange,phiRange,ZRange,print_dir,endName)

print('Generating data...')
startT_gen = time.time()

gener=generator(mRange,phiRange,ZRange,nEvents,BGtoSig=BGtoSigRatio)

gener_sigonly=generator(mRange,phiRange,ZRange,math.ceil((BGtoSigRatio[1]*nEvents)/(BGtoSigRatio[0]+BGtoSigRatio[1])),BGtoSig=(0,1))

endT_gen = time.time()
T_gen=endT_gen-startT_gen
print('Generating and preparing data took '+format(T_gen,'.2f')+'s \n\n')

Data=gener.getData()
Data_sig=gener_sigonly.getData()

print('Computing sWeights...\n')
sigWeights,bgWeights=gener.computesWeights(gener.Data[:,0])
print('\nDone.\n\n')

pter.plotSigSWeightComp(Data,sigWeights,Data_sig,plotWithErrorBars)
pter.plotSWeightedVariables(Data,sigWeights,bgWeights,plotWithErrorBars)


#need to normalise by binning!
m = np.arange(mRange[0],mRange[1],0.01)
fitres=gener.CombinedMassNExt(m,gener.sigFit[0],gener.sigFit[1],gener.bgFit[0],gener.bgFit[1],gener.bgFit[2],gener.yieldsFit[0],gener.yieldsFit[1])[1]*(0.01)
fitres_sig=gener.yieldsFit[0]*gener.SignalMassPDF(m,gener.sigFit[0],gener.sigFit[1])*(0.01)
fitres_bg=gener.yieldsFit[1]*gener.BackGPDF(m,[gener.bgFit[0],gener.bgFit[1],gener.bgFit[2]])*(0.01)


pter.plotFit(Data,m,fitres,fitres_sig,fitres_bg,False)
