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

from energyflow.archs import PFN

from sklearn.base import clone

from joblib import dump, load

def doTest(print_dir,nEvents,BGtoSig, freq=2, PhiSig=0.8, nPerfIt=[50], NTs=1,useDifData=False,bms=[GradientBoostingClassifier(max_depth=10),HistGradientBoostingClassifier(max_depth=10)],name='',doBootstrap=False):
  nEvsStr=str(int(nEvents/1000))
  endName='_'+nEvsStr+'kEvents_'+str(BGtoSig[0])+'to'+str(BGtoSig[1])+'_BGtoSig_freq'+str(freq) 
  if name!='':
    endName=name

  print('N Evs '+nEvsStr+'k BG:S '+str(BGtoSig[0])+':'+str(BGtoSig[1])+' freq '+str(freq))
  print(endName)

  mRange=(0,10)
  phiRange=(-np.pi,np.pi)
  ZRange=(-1,1)

  pter=plotter(mRange,phiRange,ZRange,print_dir,endName)

  gener=generator(mRange,phiRange,ZRange,nEvents,BGtoSig=BGtoSig,Sigma=PhiSig,frequency=freq,NTerms=NTs,generate=True,verbose=False)

  trer=trainer(bms)

  perform=performance(trer,verbose=False,useDifDataTest=useDifData)

  perfLoopSplotAll=np.zeros((1,1))
  perfLoopDRAll=np.zeros((1,1))

  totIt=0
  for nit in nPerfIt:

    #perfBTSplot = perform.do_bootstrap_splot(nit,gener)

    #perfBTDR = perform.do_bootstrap_dr(nit,gener)

    #pter.plotPerformanceResults(perfBTSplot,perfBTDR,'BootStrap')
    perfLoopSplot=np.zeros((0,0))
    perfLoopDR=np.zeros((0,0))
    if doBootstrap==False:
      perfLoopSplot, perfLoopDR = perform.do_loop(nit,gener)
    else:
      perfLoopSplot, perfLoopDR = perform.do_bootstrap_both(nit,gener)
    
    totIt=totIt+nit

    if(perfLoopSplotAll.shape[1]==1):
      perfLoopSplotAll=perfLoopSplot
      perfLoopDRAll=perfLoopDR
    else:
      perfLoopSplotAll=perform.combine_summary(perfLoopSplotAll,perfLoopSplot)
      perfLoopDRAll=perform.combine_summary(perfLoopDRAll,perfLoopDR)
    
    outName=endName
    if(len(nPerfIt)>1):
      pter.setEndName(endName+'_at'+str(totIt)+'Its')
      outName=endName+'_at'+str(totIt)+'Its'

    pter.plotPerformanceResults(perfLoopSplotAll,perfLoopDRAll,'',PhiSig)
    histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR=perform.getLoopHistos()
    pter.plotCorrelations(histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR)
    pter.plotCovariances(histoSig,histoDR,histoErrSig,histoErrDR,edges,edgesDR)

    print('\nTest Summaries at '+str(totIt)+' Iterations: ')
    #perform.print_summary(perfBTSplot,'sPlot','BootStrap')
    #perform.print_summary(perfBTDR,'DR','BootStrap')
    perform.print_summary(perfLoopSplotAll,'sPlot','Loop')
    perform.print_summary(perfLoopDRAll,'DR','Loop')

    #perform.write_out_summary(perfBTSplot,'sPlot','BootStrap','results/results'+outName+'.txt','w') ##### make sure this has option w if first !!!!
    #perform.write_out_summary(perfBTDR,'DR','BootStrap','results/results'+outName+'.txt','a')
    perform.write_out_summary(perfLoopSplotAll,'sPlot','Loop','results/results'+outName+'.txt','w') ##### make sure this has option a if not first !!!!
    perform.write_out_summary(perfLoopDRAll,'DR','Loop','results/results'+outName+'.txt','a')
    np.save('data/phi_histos'+outName+'.npy',histoSig)
    np.save('data/phi_histosDR'+outName+'.npy',histoDR)
    np.save('data/phi_histoErrs'+outName+'.npy',histoErrSig)
    np.save('data/phi_histoErrsDR'+outName+'.npy',histoErrDR)
    np.save('data/phi_edges'+outName+'.npy',edges)
    np.save('data/phi_edgesDR'+outName+'.npy',edgesDR)

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
print_dir='/w/work/clas12/tyson/plots/aiDataAcceptance/toy_DR4sWeights/tests/'

#BToSs=[ (9,1), (6,1), (4,1), (3,1), (2,1) ]
BToSs=[ (9,1) ]

# startT_it = time.time()
# for it in range(len(BToSs)):

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

  #doTest(print_dir,100000,BToSs[it],freq=2,nPerfIt=[50])


nEvs=[750000,1000000] #100000 already in previous test  1000, 10000 ,200000,500000,#


# startT_it = time.time()
# for it in range(len(nEvs)):
#   bm=[GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]
#   nEvStr=str(math.ceil(nEvs[it]/1000.))
#   endName='_'+nEvStr+'kEvents_9to1_BGtoSig_freq2_model_GBDTandGBDT'

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

  #doTest(print_dir,nEvs[it],(9,1),freq=2,bms=bm,nPerfIt=[50],name=endName)

freqs=[1,2,4,8,10] #1,2,6
# freqs=[2]
#freqs=[4,8,10]
#freqs=[10]

# startT_it = time.time()
# for it in range(len(freqs)):
#   bm=[GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]
#   #bm=['NN']
#   endName='_100kEvents_9to1_BGtoSig_freq'+str(freqs[it])+'_capv2_model_2GBT'

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

#   doTest(print_dir,100000,(9,1),freq=freqs[it],bms=bm,nPerfIt=[50],name=endName)


# startT_it = time.time()
# for it in range(len(freqs)):
#   bm=[GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]
#   #bm=['NN']
#   endName='_100kEvents_2to1_BGtoSig_freq'+str(freqs[it])+'_capv2_model_2GBT'

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

#   doTest(print_dir,100000,(2,1),freq=freqs[it],bms=bm,nPerfIt=[50],name=endName)
    


    
#bmst=[['PFN'],['PFN',GradientBoostingClassifier(max_depth=10)],['PFN',HistGradientBoostingClassifier(max_depth=10)]]
#modelnames=['PFN2','PFN2andGBDT','PFN2andHistGBDT']

#bmst=[['NN'],['NN',GradientBoostingClassifier(max_depth=10)],['NN',HistGradientBoostingClassifier(max_depth=10)]]
#modelnames=['NN','NNandGBDT','NNandHistGBDT']

bmst=[['NN',GradientBoostingClassifier(max_depth=10)],['NN',HistGradientBoostingClassifier(max_depth=10)]]
modelnames=['NNandGBDT','NNandHistGBDT']


#bmst=[[GradientBoostingClassifier(max_depth=10)], [HistGradientBoostingClassifier(max_depth=10)], [ExtraTreesClassifier(max_features=None,criterion='log_loss')], [GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)], [HistGradientBoostingClassifier(max_depth=10),HistGradientBoostingClassifier(max_depth=10)], [HistGradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]]
#modelnames=['GBDT','HistGDBT','ERT','GBDTandGBDT','HistGBDTandHistGBDT','HistGBDTandGBDT']
    
# startT_it = time.time()
# for it in range(len(bmst)):
#   endName='_100kEvents_9to1_BGtoSig_freq2_model_'+modelnames[it]

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

  #doTest(print_dir,100000,(9,1),freq=2,nPerfIt=[50],bms=bmst[it],name=endName)

NTerms=[6] #[1,2,3,4,5]

# startT_it = time.time()
# for it in range(len(NTerms)):
#   bm=[GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]
#   endName='_100kEvents_2to1_BGtoSig_freq2_model_GBDTandGBDT_Sigma0p15_NTerms'+str(NTerms[it])

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')


#  doTest(print_dir,100000,(2,1),PhiSig=0.15,freq=2,bms=bm,nPerfIt=[50],NTs=NTerms[it],name=endName)

# nGBT=[1,2,5,10,25]#,50] #2 sometimes done above
#nGBT=[25,10,5]
nGBT=[25]

startT_it = time.time()
for it in range(len(nGBT)):
  bm=[]
  for n in range(nGBT[it]):
    bm.append(GradientBoostingClassifier(max_depth=10)) #10
  endName='_100kEvents_9to1_BGtoSig_freq10_capv2_at1_model_'+str(nGBT[it])+'GBT' #_bootstrapv2'

  if it==0:
    print('!!!!!! performing test iteration :',it,' !!!!!')
  else:
    endT_it = time.time()
    T_it=(endT_it-startT_it)
    print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

  doTest(print_dir,100000,(9,1),freq=10,bms=bm,nPerfIt=[50,50,400,500,1000],name=endName,useDifData=False,doBootstrap=False) #nPerfIt=[50,50,400,500]

depth=[3,5,10,25,50,100]#,50] #2 sometimes done above

# startT_it = time.time()
# for it in range(len(nGBT)):
#   bm=[GradientBoostingClassifier(max_depth=depth[it])]
#   endName='_100kEvents_2to1_BGtoSig_freq2_capv2_at1_model_1GBT_depth'+str(depth[it])

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

#   doTest(print_dir,100000,(2,1),freq=2,bms=bm,nPerfIt=[50],name=endName,useDifData=False)

nDifData=[False,True] #[5,10,25,50]

# startT_it = time.time()
# for it in range(len(nDifData)):
#   bm=[GradientBoostingClassifier(max_depth=10),GradientBoostingClassifier(max_depth=10)]
#   endName='_100kEvents_9to1_BGtoSig_freq10_capAll0p75_model_2GBT'
#   if nDifData[it]==True:
#     endName=endName+'_DifTestDataTrue'
#   else:
#     endName=endName+'_DifTestDataFalse'

#   if it==0:
#     print('!!!!!! performing test iteration :',it,' !!!!!')
#   else:
#     endT_it = time.time()
#     T_it=(endT_it-startT_it)
#     print('\n\n!!!!!!! performing test iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s !!!!!!')

#   doTest(print_dir,100000,(9,1),freq=10,bms=bm,nPerfIt=[50],useDifData=nDifData[it],name=endName)
    


endT_all = time.time()
T_all=(endT_all-startT_all)/60

print('\nEntire script took '+format(T_all,'.2f')+' minutes\n')

