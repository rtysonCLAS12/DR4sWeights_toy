import numpy as np
import time as timeCount
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import math
import time

from plotter import plotter

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

nIts=[50,100,500,1000,2000]
for it in range(len(nIts)):
  bm=[]
  endName='_100kEvents_9to1_BGtoSig_freq10_capv2_at1_model_2GBT' #_bootstrapv2'
  endName=endName+'_at'+str(nIts[it])+'Its'

  print('Plotting iteration '+str(nIts[it]))

  mRange=(0,10)
  phiRange=(-np.pi,np.pi)
  ZRange=(-1,1)

  pter=plotter(mRange,phiRange,ZRange,print_dir,endName)

  histoSig=np.load('data/phi_histos'+endName+'.npy')
  histoDR=np.load('data/phi_histosDR'+endName+'.npy')
  histoErrSig=np.load('data/phi_histoErrs'+endName+'.npy')
  histoErrDR=np.load('data/phi_histoErrsDR'+endName+'.npy')
  edges=np.load('data/phi_edges'+endName+'.npy')
  edgesDR=np.load('data/phi_edgesDR'+endName+'.npy')

  #pter.plotCorrelations(histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR)
  pter.plotCovariances(histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR)

  if nIts[it]==2000:
    pter.plotPulls(histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR)