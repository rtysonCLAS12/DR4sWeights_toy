from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import mplhep
import numpy as np

class plotter:
  print_dir=''
  endName=''
  ranges=[]

  names=[]
  titles=[]
  units=[]

  def __init__(self,mR,phiR,zR,pd='',en=''):
    self.ranges.append(mR)
    self.ranges.append(phiR)
    self.ranges.append(zR)
    self.ranges.append((-1,1)) #to plot weights
    self.print_dir=pd
    self.endName=en

    self.names=['_M','_Phi','_Z','_weights']
    self.titles=['Mass',r'$\phi$','Z','Weights']
    self.units=['[GeV]','[rad]','','']


  def plotSWeightedVariables(self,vars,sWeights,sWeightsBG,useErrorBars=False):

    for j in range(3):

      if useErrorBars==False:
        fig = plt.figure(figsize=(20, 20))
        plt.hist(vars[:,j], range=self.ranges[j],bins=100,color='royalblue',label='sWeights Signal',weights=sWeights)
        plt.hist(vars[:,j], range=self.ranges[j],bins=100,edgecolor='firebrick',label='sWeights Background',hatch='/', histtype='step',fill=False,linewidth=3,weights=sWeightsBG)
        plt.hist(vars[:,j], range=self.ranges[j],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
        plt.legend(loc='upper right')
        ymin, ymax = plt.ylim()
        if ymin<0:
          plt.ylim(ymin, ymax * 1.25)
        else:
          plt.ylim(0, ymax * 1.25)
        plt.xlabel(self.titles[j]+' '+self.units[j])
        plt.title(self.titles[j])
        plt.savefig(self.print_dir+'sWeights'+self.names[j]+self.endName+'.png')

      else:
        fig = plt.figure(figsize=(20, 20))
        nsall, ball = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100)
        mplhep.histplot(nsall, bins=ball, histtype="errorbar", yerr=True,label="All", color="black",linewidth=3,markersize=25,capsize=7,elinewidth=5)
        nssig, bsig = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=sWeights)
        mplhep.histplot(nssig, bins=bsig, histtype="errorbar", yerr=True,label="sWeights Signal", color="royalblue",linewidth=3,markersize=25,capsize=7,elinewidth=5)
        nsbg, bbg = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=sWeightsBG)
        mplhep.histplot(nsbg, bins=bbg, histtype="errorbar", yerr=True,label="sWeights Background", color="firebrick",linewidth=3,markersize=25,capsize=7,elinewidth=5)

        plt.legend(loc='upper right')
        ymin, ymax = plt.ylim()
        if ymin<0:
          plt.ylim(ymin, ymax * 1.25)
        else:
          plt.ylim(0, ymax * 1.25)
        plt.xlabel(self.titles[j]+' '+self.units[j])
        plt.title(self.titles[j])
        plt.savefig(self.print_dir+'sWeights'+self.names[j]+self.endName+'.png')

  def plotWeights(self,weightsSig,weightsDR,useErrorBars=False):
    j=3
    if useErrorBars==False:
      fig = plt.figure(figsize=(20, 20))
      plt.hist(weightsSig, range=self.ranges[j],bins=100,color='royalblue',label='sWeights Signal')
      plt.hist(weightsDR, range=self.ranges[j],bins=100,edgecolor='firebrick',label='Density Ratio Signal',hatch='/', histtype='step',fill=False,linewidth=3)
      plt.legend(loc='upper right')
      plt.yscale("log")   
      ymin, ymax = plt.ylim()
      plt.ylim(0.01, ymax * 10)
      plt.xlabel(self.titles[j]+' '+self.units[j])
      plt.title(self.titles[j])
      plt.savefig(self.print_dir+'Comp'+self.names[j]+self.endName+'.png')
    else:
      fig = plt.figure(figsize=(20, 20))
      nssig, bsig = np.histogram(weightsSig,range=(self.ranges[j][0],self.ranges[j][1]), bins=100)
      mplhep.histplot(nssig, bins=bsig, histtype="errorbar", yerr=True,label="sWeights Signal", color="royalblue",linewidth=3,markersize=25,capsize=7,elinewidth=5)
      nsbg, bbg = np.histogram(weightsDR,range=(self.ranges[j][0],self.ranges[j][1]), bins=100)
      mplhep.histplot(nsbg, bins=bbg, histtype="errorbar", yerr=True,label="Density Ratio Signal", color="firebrick",linewidth=3,markersize=25,capsize=7,elinewidth=5)

      plt.legend(loc='upper right')
      plt.yscale("log")   
      ymin, ymax = plt.ylim()
      plt.ylim(10, ymax * 10)
      plt.xlabel(self.titles[j]+' '+self.units[j])
      plt.title(self.titles[j])
      plt.savefig(self.print_dir+'Comp'+self.names[j]+self.endName+'.png')


  def plotDRToSWeightComp(self,vars,sWeights,DRWeights,useErrorBars=False):

    for j in range(1,3):

      nall=[]
      ball=[]
      pall=[]
      nsig=[]
      bsig=[]
      psig=[]
      npred=[]
      bpred=[]
      ppred=[]

      fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True, sharey=False,tight_layout=True,height_ratios=[2, 1])
      if useErrorBars==False:
        nsig,bsig,psig=axs[0].hist(vars[:,j], range=self.ranges[j],bins=100,color='royalblue',label='sWeights Signal',weights=sWeights)
        npred,bpred,ppred=axs[0].hist(vars[:,j], range=self.ranges[j],bins=100,edgecolor='firebrick',label='Density Ratio Signal',hatch='/', histtype='step',fill=False,linewidth=3,weights=DRWeights)
        nall,ball,pall=axs[0].hist(vars[:,j], range=self.ranges[j],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
        axs[0].legend(loc='upper right')
        ymin, ymax = axs[0].get_ylim()
        if ymin<0:
          axs[0].set_ylim(ymin, ymax * 1.35)
        else:
          axs[0].set_ylim(0, ymax * 1.35)
        axs[0].set_title(self.titles[j])

      else:
        #nall, ball = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100)
        #mplhep.histplot(nall, bins=ball, histtype="errorbar", yerr=True,label="All", color="black",linewidth=3,ax=axs[0],markersize=25,capsize=7,elinewidth=5)
        #nsig, bsig = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=sWeights)
        #mplhep.histplot(nsig, bins=bsig, histtype="errorbar", yerr=True,label="sWeights Signal", color="royalblue",linewidth=3,ax=axs[0],markersize=25,capsize=7,elinewidth=5)
        #npred, bpred = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=DRWeights)
        #mplhep.histplot(npred, bins=bpred, histtype="errorbar", yerr=True,label="Density Ratio Signal", color="firebrick",linewidth=3,ax=axs[0],markersize=25,capsize=7,elinewidth=5)

        hall,nall,errall,ball=self.histWeightedErrorBars(vars[:,j], np.ones((vars.shape[0])), np.ones((vars.shape[0])), color="black", label="All",ax=axs[0],rge=(self.ranges[j][0],self.ranges[j][1]))
        hsig,nsig,errsig,bsig=self.histWeightedErrorBars(vars[:,j], sWeights, sWeights, color="royalblue", label="sWeights Signal",ax=axs[0],rge=(self.ranges[j][0],self.ranges[j][1]))
        hpred,npred,errpred,bpred=self.histWeightedErrorBars(vars[:,j], DRWeights, sWeights, color="firebrick", label="Density Ratio Signal",ax=axs[0],rge=(self.ranges[j][0],self.ranges[j][1]))


        axs[0].legend(loc='upper right')
        ymin, ymax = axs[0].get_ylim()
        if ymin<0:
          axs[0].set_ylim(ymin, ymax * 1.35)
        else:
          axs[0].set_ylim(0, ymax * 1.35)
        axs[0].set_title(self.titles[j])


      res=[]
      res_err=[]
      plotloc=[]
      for i in range(len(nsig)):
        if (npred[i]!=0) or (nsig[i]!=0):
          #res.append(nsig[i]/npred[i])
          #e=res[i]*np.sqrt( np.square((np.sqrt(nsig[i])/nsig[i])) +  np.square((np.sqrt(npred[i])/npred[i])) ) #error for ratio
          std=np.sqrt(errsig[i]**2 + errpred[i]**2) #sum of squares of sqrt(nsig) and sqrt(npred)
          res.append((nsig[i]-npred[i])/std)
          res_err.append(1)
          plotloc.append( (bsig[i+1]-bsig[i])/2 + bsig[i] )
        
      axs[1].errorbar(x=plotloc, y=res, yerr=res_err,fmt='s', color='mediumorchid',ms=10,capsize=7,elinewidth=3)
      axs[1].axhline(y = 0.0, color = 'black', linestyle = '--')#,label='1') 
      axs[1].axhline(y = -1.0, color = 'grey', linestyle = '--')#,label='1') 
      axs[1].axhline(y = 1.0, color = 'grey', linestyle = '--')#,label='1') 
      axs[1].axhline(y = -2.0, color = 'silver', linestyle = '--')#,label='1') 
      axs[1].axhline(y = 2.0, color = 'silver', linestyle = '--')#,label='1') 
      axs[1].set_ylim(-5, 5)
      axs[1].set_ylabel('Pull')
      plt.xlabel(self.titles[j]+' '+self.units[j])
      plt.savefig(self.print_dir+'DRsWeights_Comp'+self.names[j]+self.endName+'.png')

  def histWeightedErrorBars(self,data, wgts, errwgts, color, label,ax,rge):
    sumweights, edges = np.histogram( data, weights=wgts, range=rge,bins=100 )
    sumweight_sqrd, edges = np.histogram( data, weights=errwgts*errwgts, range=rge,bins=100 )
    errs = np.sqrt(sumweight_sqrd)
    bincenters = []
    for i in range(len(sumweights)):
      bincenters.append( (edges[i+1]-edges[i])/2 + edges[i] )
    histo = ax.errorbar(bincenters, sumweights, errs, fmt='.',color=color, label=label,linewidth=3,markersize=25,capsize=7,elinewidth=5)
    return histo,sumweights,errs,edges

  def plotPerformanceResults(self,perf,perfDR,testName):

    title=['Asymmetry','Asymmetry Uncertainty','Mean of Pulls','Standard Deviation of Pulls',r'$\chi^{2}$']
    name=['asym','asymErr','pull','stdpull','chi2']
    ranges=[(0.65,0.95),(0,0.2),(-0.1,0.1),(0,5),(0.0,5)]

    for i in range(perf.shape[1]):
      fig = plt.figure(figsize=(20, 20))
      plt.hist(perf[:,i], range=ranges[i],bins=20,color='royalblue',label='sWeights Signal')
      plt.hist(perfDR[:,i], range=ranges[i],bins=20,edgecolor='firebrick',label='Density Ratio Signal',hatch='/', histtype='step',fill=False,linewidth=3)
      plt.legend(loc='upper right')
      ymin, ymax = plt.ylim()
      plt.ylim(0, ymax * 1.25)
      plt.xlabel(title[i])
      if testName=='':
        plt.title(title[i])
        plt.savefig(self.print_dir+'loop_'+name[i]+self.endName+'.png')
      else:
        plt.title(title[i]+' ('+testName+')')
        plt.savefig(self.print_dir+testName+'_'+name[i]+self.endName+'.png')
      
