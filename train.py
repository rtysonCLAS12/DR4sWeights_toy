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

from joblib import dump, load

def rejection_sample(weights):
    nev=weights.shape[0]
    rands = np.random.uniform(0, 1, nev)
    mask = rands[:] < weights[:]
    return mask.astype(int) 

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

startT_all = time.time()

#plots outputted to print_dir
#append endName to end of plots to avoid overwriting plots
print_dir='/w/work/clas12/tyson/plots/aiDataAcceptance/toy_DR4sWeights/vars/'
endName='_test' 

plotWithErrorBars=True
doFineTune=True

nEvents=100000
#nEvents=10000

mRange=(0,10)
phiRange=(-np.pi,np.pi)
ZRange=(-1,1)

print('\nPlots will be written to directory: '+print_dir)
print('Plot names formatted as name'+endName+'.pdf\n\n')

pter=plotter(mRange,phiRange,ZRange,print_dir,endName)

print('Generating data...')
startT_gen = time.time()

gener=generator(mRange,phiRange,ZRange,nEvents)

endT_gen = time.time()
T_gen=endT_gen-startT_gen
print('Generating and preparing data took '+format(T_gen,'.2f')+'s \n\n')

Data=gener.getData()

print('Computing sWeights...\n')
sigWeights,bgWeights=gener.computesWeights(gener.Data[:,0])
print('\nDone.\n\n')

#pter.plot_fit_projection(gener.sPlotModel, gener.sPlotData, nbins=100)
pter.plotSWeightedVariables(Data,sigWeights,bgWeights,plotWithErrorBars)

sigWeights=np.asarray(sigWeights)#.reshape((Data.shape[0],1))

Data=gener.scale(Data)

#split into training and testing sets
nTrain=math.ceil(0.7*(Data.shape[0]))
X_train=Data[:nTrain,:]
X_test=Data[nTrain:,:]

weights_train=sigWeights[:nTrain]#.reshape((X_train))
weights_test=sigWeights[nTrain:]#.reshape((X_test))

#both GradientBoosting and HistGradientBoosting work well
#HistGradientBoosting is much faster

#base_model = HistGradientBoostingClassifier(max_depth=10)
base_model = GradientBoostingClassifier(max_depth=10)
#base_model = ExtraTreesClassifier(max_features=None,criterion='log_loss')

base_model2 = HistGradientBoostingClassifier(max_depth=10)
#base_model2 = GradientBoostingClassifier(max_depth=10)
#base_model2 = ExtraTreesClassifier(max_features=None,criterion='log_loss')

bms=[base_model,base_model2]

trer=trainer(bms)

print('Training with '+str(X_train.shape[0]*2)+' events...')

trer.train(X_train,weights_train)

print('Test with '+str(X_test.shape[0])+' events...')

weights_DR=trer.predict(X_test)

X_test=gener.unscale(X_test)

pter.plotDRToSWeightComp(X_test,weights_test,weights_DR,plotWithErrorBars)

print('\nFitting sWeighted Asymmetry')
gener.fitAsymmetry(X_test,weights_test,weights_test*weights_test)

print('\nFitting Density Ratio Weighted Asymmetry')
gener.fitAsymmetry(X_test,weights_DR,weights_test*weights_test)


nPerfIt=50

#performance assumes unscaled data!!
Data=gener.unscale(Data)

perform=performance(trer)
startT_boot = time.time()
print('\n\nBootStrap sWeighted Asymmetry\n\n')

perfBTSplot = perform.do_bootstrap_splot(nPerfIt,gener)

print('\n\nBootStrap drWeighted Asymmetry\n\n')

perfBTDR = perform.do_bootstrap_dr(nPerfIt,gener)

endT_boot = time.time()
T_boot=(endT_boot-startT_boot)/60
print('\nBootstraps took '+format(T_boot,'.2f')+' minutes\n')

pter.plotPerformanceResults(perfBTSplot,perfBTDR,'BootStrap')

startT_loop = time.time()
print('\n\nLoop sWeighted Asymmetry\n\n')

perfLoopSplot, perfLoopDR = perform.do_loop(nPerfIt,gener)

endT_loop = time.time()
T_loop=(endT_loop-startT_loop)/60
print('\nLoops took '+format(T_loop,'.2f')+' minutes\n')

pter.plotPerformanceResults(perfLoopSplot,perfLoopDR,'')

print('\n\n\n\n Test Summaries:')
perform.print_summary(perfBTSplot,'sPlot','BootStrap')
perform.print_summary(perfBTDR,'DR','BootStrap')
perform.print_summary(perfLoopSplot,'sPlot','Loop')
perform.print_summary(perfLoopDR,'DR','Loop')


endT_all = time.time()
T_all=(endT_all-startT_all)/60

print('\nEntire script took '+format(T_all,'.2f')+' minutes\n')

