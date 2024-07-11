from generator import generator
from plotter import plotter

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

startT_all = time.time()

#plots outputted to print_dir
#append endName to end of plots to avoid overwriting plots
print_dir=''
endName='_HistGBDTs' 

plotWithErrorBars=False

nEvents=1000000

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
sigWeights,bgWeights=gener.computesWeights()
print('\nDone.\n\n')

#pter.plot_fit_projection(gener.sPlotModel, gener.sPlotData, nbins=100)
pter.plotSWeightedVariables(Data,sigWeights,bgWeights,plotWithErrorBars)

sigWeights=np.asarray(sigWeights).reshape((Data.shape[0],1))

Data=gener.scale(Data)

#training data is composed of twice the data
#weighted with sWeights and by one
Xall=np.vstack((Data,Data))
weights=np.vstack((sigWeights,np.ones((Data.shape[0],1)))).reshape((Xall.shape[0]))
Yall=np.vstack((np.ones((Data.shape[0],1)),np.zeros((Data.shape[0],1)))).reshape((Xall.shape[0]))

#shuffle in unison
p = np.random.permutation(Xall.shape[0])
Xall=Xall[p]
Yall=Yall[p]
weights=weights[p]

#split into training and testing sets
nTrain=math.ceil(0.7*(Xall.shape[0]))
X_train=Xall[:nTrain,:]
X_test=Xall[nTrain:,:]

y_train=Yall[:nTrain]
y_test=Yall[nTrain:]

weights_train=weights[:nTrain]
weights_test=weights[nTrain:]

#both GradientBoosting and HistGradientBoosting work well
#HistGradientBoosting is much faster

model = HistGradientBoostingClassifier(max_depth=10)
#model = GradientBoostingClassifier(max_depth=10)

print('Training with '+str(X_train.shape[0])+' events...')

#train model
startT_train = time.time()

#don't include mass at var 0 in fit
model.fit(X_train[:,1:],y_train,sample_weight=weights_train)

endT_train = time.time()
T_train=(endT_train-startT_train)/60

print('Training took '+format(T_train,'.2f')+' minutes\n')

#save model
dump(model,'model'+endName+'.joblib')

#to compare to sWeights we want to only predict
#on the sWeighted subset of the training data
X_test=X_test[y_test==1]
weights_test=weights_test[y_test==1]
y_test=y_test[y_test==1]

print('Test with '+str(X_test.shape[0])+' events...')

#test model
startT_test = time.time()

y_pred=model.predict_proba(X_test[:,1:])[:,1]

endT_test = time.time()
T_test=(endT_test-startT_test)

print('Testing took '+format(T_test,'.4f')+' seconds\n')

#we can now calculate the Density Ratio estimated Weights
y_pred[y_pred==1]=1-0.0000001
weights_DR = y_pred/(1-y_pred)
weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
weights_DR[weights_DR>1]=1 #some weights blow up

X_test=gener.unscale(X_test)

pter.plotDRToSWeightComp(X_test,weights_test,weights_DR,plotWithErrorBars)

print('\nFitting sWeighted Asymmetry')
gener.fitAsymmetry(X_test,weights_test,weights_test*weights_test)

print('\nFitting Density Ratio Weighted Asymmetry')
gener.fitAsymmetry(X_test,weights_DR,weights_test*weights_test)

endT_all = time.time()
T_all=(endT_all-startT_all)/60

print('\nEntire script took '+format(T_all,'.2f')+' minutes\n')