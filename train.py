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

plt.rcParams.update({'font.size': 30,
                    #'font.family':  'Times New Roman',
                    'legend.edgecolor': 'black',
                    'xtick.minor.visible': True,
                    'ytick.minor.visible': True,
                    'xtick.major.size':15,
                    'xtick.minor.size':10,
                    'ytick.major.size':15,
                     'ytick.minor.size':10,
                     'figure.max_open_warning':200})

startT_all = time.time()

#plots outputted to print_dir
#append endName to end of plots to avoid overwriting plots
print_dir=''
endName='_HistGBDTs' 

noP_inTrain=True

trainvars=[0,1,2,3,4,5]
if noP_inTrain:
  trainvars=[1,2,4,5]

masses=np.array([0.1395703,0.1395703])#np.array([0.,0.])
targetMass=0.777#0.1349768
nSignalEvents=100000
nBGEvents=100000

print('\nPlots will be written to directory: '+print_dir)
print('Plot names formatted as name'+endName+'.pdf\n\n')

pter=plotter(targetMass,print_dir,endName)

print('Generating data...')
startT_gen = time.time()

gener=generator(masses,targetMass,nSignalEvents,nBGEvents)

endT_gen = time.time()
T_gen=endT_gen-startT_gen
print('Generating and preparing data took '+format(T_gen,'.2f')+'s \n\n')

DataSig, IMSig=gener.getSignal()
DataBG, IMBG=gener.getBackground()
DataAll, IMAll=gener.getMixedData()

pter.plotIMComparison(IMSig,IMBG,IMAll)

print('Computing sWeights...\n')
sWeights=gener.computesWeights()
print('\nDone.\n\n')

pter.plot_fit_projection(gener.sPlotModel, gener.sPlotData, nbins=100)
pter.plotSWeightedVariables(np.hstack((DataAll,IMAll.reshape((IMAll.shape[0],1)))),sWeights.sig,sWeights.bck,len(masses))

signalSWeights=(sWeights.sig).to_numpy().reshape((DataAll.shape[0],1))


#training data is composed of twice the data
#weighted with sWeights and by one
X=np.vstack((DataAll,DataAll))
weights=np.vstack((signalSWeights,np.ones((DataAll.shape[0],1)))).reshape((X.shape[0]))
Y=np.vstack((np.ones((DataAll.shape[0],1)),np.zeros((DataAll.shape[0],1)))).reshape((X.shape[0]))

#shuffle in unison
p = np.random.permutation(X.shape[0])
X=X[p]
Y=Y[p]
weights=weights[p]

#split into training and testing sets
nTrain=math.ceil(0.7*(X.shape[0]))
X_train=X[:nTrain,:]
X_test=X[nTrain:,:]

y_train=Y[:nTrain]
y_test=Y[nTrain:]

weights_train=weights[:nTrain]
weights_test=weights[nTrain:]

#both GradientBoosting and HistGradientBoosting work well
#HistGradientBoosting is much faster

model = HistGradientBoostingClassifier(max_depth=10)
#model = GradientBoostingClassifier(max_depth=10)

print('Training with '+str(X_train.shape[0])+' events...')

#train model
startT_train = time.time()

model.fit(X_train[:,trainvars],y_train,sample_weight=weights_train)

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

y_pred=model.predict_proba(X_test[:,trainvars])[:,1]

endT_test = time.time()
T_test=(endT_test-startT_test)

print('Testing took '+format(T_test,'.4f')+' seconds\n')

#we can now calculate the Density Ratio estimated Weights
y_pred[y_pred==1]=1-0.0000001
weights_DR = y_pred/(1-y_pred)
weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
weights_DR[weights_DR>1]=1 #some weights blow up

pter.plotDRToSWeightComp(X_test,weights_test,weights_DR,len(masses))

endT_all = time.time()
T_all=(endT_all-startT_all)/60

print('\nEntire script took '+format(T_all,'.2f')+' minutes\n')