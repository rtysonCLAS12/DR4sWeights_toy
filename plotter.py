from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm, SymLogNorm
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

  def setEndName(self, en):
    self.endName=en

  def plotSigSWeightComp(self,vars,sWeights,sigOnly,useErrorBars=False):

    for j in range(3):

      if useErrorBars==False:
        fig = plt.figure(figsize=(20, 20))
        plt.hist(vars[:,j], range=self.ranges[j],bins=100,color='royalblue',label='sWeighted Signal',weights=sWeights)
        plt.hist(sigOnly[:,j], range=self.ranges[j],bins=100,edgecolor='mediumorchid',label='Signal',hatch='/', histtype='step',fill=False,linewidth=3,weights=np.ones((sigOnly.shape[0])))
        plt.hist(vars[:,j], range=self.ranges[j],bins=100,edgecolor='black',color='black',label='Signal & Background', histtype='step',fill=False,linewidth=3)
        plt.legend(loc='upper right')
        ymin, ymax = plt.ylim()
        if ymin<0:
          plt.ylim(ymin, ymax * 1.25)
        else:
          plt.ylim(0, ymax * 1.25)
        plt.xlabel(self.titles[j]+' '+self.units[j])
        plt.title(self.titles[j])
        plt.savefig(self.print_dir+'sWeightSignalComp'+self.names[j]+self.endName+'.png')

      else:
        fig = plt.figure(figsize=(20, 20))
        nsall, ball = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100)
        mplhep.histplot(nsall, bins=ball, histtype="errorbar", yerr=True,label="Signal & Background", color="black",linewidth=3,markersize=25,capsize=7,elinewidth=5)
        nssig, bsig = np.histogram(vars[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=sWeights)
        mplhep.histplot(nssig, bins=bsig, histtype="errorbar", yerr=True,label="sWeighted Signal", color="royalblue",linewidth=3,markersize=25,capsize=7,elinewidth=5)
        nsbg, bbg = np.histogram(sigOnly[:,j],range=(self.ranges[j][0],self.ranges[j][1]), bins=100,weights=np.ones((sigOnly.shape[0])))
        mplhep.histplot(nsbg, bins=bbg, histtype="errorbar", yerr=True,label="Signal", color="mediumorchid",linewidth=3,markersize=25,capsize=7,elinewidth=5)

        plt.legend(loc='upper right')
        ymin, ymax = plt.ylim()
        if ymin<0:
          plt.ylim(ymin, ymax * 1.25)
        else:
          plt.ylim(0, ymax * 1.25)
        plt.xlabel(self.titles[j]+' '+self.units[j])
        plt.title(self.titles[j])
        plt.savefig(self.print_dir+'sWeightSignalComp'+self.names[j]+self.endName+'.png')

  def plotFit(self,vars,m,Fit,Fit_sig,Fit_bg,useErrorBars=False):

    if useErrorBars==False:
      fig = plt.figure(figsize=(20, 20))
      plt.hist(vars[:,0], range=self.ranges[0],bins=100,edgecolor='black',color='black',label='Signal & Background', histtype='step',fill=False,linewidth=5)
      plt.plot(m, Fit_sig, label="Signal Fit", color="royalblue",linewidth=5,linestyle='--')
      plt.plot(m, Fit_bg, label="Bg Fit", color="firebrick",linewidth=5,linestyle='--')
      plt.plot(m, Fit, label="Total Fit", color="mediumorchid",linewidth=5)
      #plt.legend(loc='upper right')
      #ymin, ymax = plt.ylim()
      #if ymin<0:
      #  plt.ylim(ymin, ymax * 1.25)
      #else:
      #  plt.ylim(0, ymax * 1.25)
      plt.xlabel(self.titles[0]+' '+self.units[0])
      plt.title(self.titles[0])
      plt.savefig(self.print_dir+'Fit'+self.names[0]+self.endName+'.png')
      
    else:
      fig = plt.figure(figsize=(20, 20))
      nsall, ball = np.histogram(vars[:,0],range=(self.ranges[0][0],self.ranges[0][1]), bins=100)
      mplhep.histplot(nsall, bins=ball, histtype="errorbar", yerr=True,label="Signal & Background", color="black",linewidth=3,markersize=25,capsize=7,elinewidth=5)
      plt.plot(m, Fit_sig, label="Signal Fit", color="royalblue",linewidth=5,linestyle='--')
      plt.plot(m, Fit_bg, label="Bg Fit", color="firebrick",linewidth=5,linestyle='--')
      plt.plot(m, Fit, label="Total Fit", color="mediumorchid",linewidth=5)
      #plt.legend(loc='upper right')
      #ymin, ymax = plt.ylim()
      #if ymin<0:
      #  plt.ylim(ymin, ymax * 1.25)
      #else:
      #  plt.ylim(0, ymax * 1.25)
      plt.xlabel(self.titles[0]+' '+self.units[0])
      plt.title(self.titles[0])
      plt.savefig(self.print_dir+'Fit'+self.names[0]+self.endName+'.png')


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
        axs[0].axhline(y=0, color='grey', linestyle='--')



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
    sumweights, edges = np.histogram( data, weights=wgts, range=rge,bins=100)
    sumweight_sqrd, edges = np.histogram( data, weights=errwgts*errwgts, range=rge,bins=100)
    errs = np.sqrt(sumweight_sqrd)
    bincenters = []
    for i in range(len(sumweights)):
      bincenters.append( (edges[i+1]-edges[i])/2 + edges[i] )
    histo = ax.errorbar(bincenters, sumweights, errs, fmt='.',color=color, label=label,linewidth=3,markersize=25,capsize=7,elinewidth=5)
    return histo,sumweights,errs,edges

  def plotPerformanceResults(self,perf,perfDR,testName,Sigma):

    title=['Asymmetry','Asymmetry Uncertainty','Mean of Pulls','Standard Deviation of Pulls',r'$\chi^{2}$']
    name=['asym','asymErr','pull','stdpull','chi2']
    ranges=[(Sigma-0.20,Sigma+0.20),(0,0.2),(-0.1,0.1),(0,5),(0.0,2.0)]

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

  def plotCorrelations(self,h,hDR,herr,herrDR,edges,edgesDR):
    # co_sig = np.cov(h, rowvar=False)
    # co_DR = np.cov(hDR, rowvar=False)
    co_sig = np.corrcoef(h, rowvar=False)
    co_DR = np.corrcoef(hDR, rowvar=False)
    #print(h.shape)

    edg=edges[0]
    bin_centers = 0.5 * (edg[:-1] + edg[1:])
    num_ticks = 7
    #tick_indices = np.linspace(0, len(bin_centers) - 1, num=num_ticks, dtype=int)
    #tick_labels = [f"{float(bin_centers[i]):.2f}" for i in tick_indices]
    tick_indices = np.linspace(0, len(edg)-1, 7, dtype=int)
    tick_positions = [edg[i] for i in tick_indices]
    tick_labels = [f"{edg[i]:.2f}" for i in tick_indices]

    extent = [edg[0], edg[-1], edg[0], edg[-1]]

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(35, 17))

    im1 = axs[0].imshow(co_sig, cmap='coolwarm', interpolation='none', origin='lower', vmin=-1, vmax=1, extent=extent, aspect='auto')
    axs[0].set_title('sWeights Signal')
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(co_DR, cmap='coolwarm', interpolation='none', origin='lower', vmin=-1, vmax=1, extent=extent, aspect='auto')
    axs[1].set_title('Density Ratio Signal')
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
      ax.set_xlabel(r'$\phi$ [rad]')
      ax.set_ylabel(r'$\phi$ [rad]')
      ax.set_xticks(tick_positions)
      ax.set_xticklabels(tick_labels) #, rotation=90)
      ax.set_yticks(tick_positions)
      ax.set_yticklabels(tick_labels)

    fig.suptitle('Bin-to-Bin Correlation Matrices')
    plt.tight_layout()
    plt.savefig(self.print_dir+'BinsCorr_Comp_phi'+self.endName+'.png')

  def plotCovariances(self,h,hDR,herr,herrDR,edges,edgesDR):
    co_sig = np.cov(h, rowvar=False)
    co_DR = np.cov(hDR, rowvar=False)
    #print(h.shape)

    max_sig=np.amax(co_sig)
    max_dr=np.amax(co_DR)
    min_sig=np.amin(co_sig)
    min_dr=np.amin(co_DR)
    max=max_dr+0.05*max_dr
    min=min_dr-0.05*min_dr
    if max_sig>max_dr:
      max=max_sig+0.05*max_sig

    if min_sig<min_dr:
      min=min_sig-0.05*min_sig

    if min<0:
      if np.abs(max)>np.abs(min):
        min=-max
      else:
        max=-min
    else:
      min=-max

    max=300
    min=-300


    edg=edges[0]
    bin_centers = 0.5 * (edg[:-1] + edg[1:])
    num_ticks = 7
    #tick_indices = np.linspace(0, len(bin_centers) - 1, num=num_ticks, dtype=int)
    #tick_labels = [f"{float(bin_centers[i]):.2f}" for i in tick_indices]
    tick_indices = np.linspace(0, len(edg)-1, 7, dtype=int)
    tick_positions = [edg[i] for i in tick_indices]
    tick_labels = [f"{edg[i]:.2f}" for i in tick_indices]

    extent = [edg[0], edg[-1], edg[0], edg[-1]]

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(35, 17))

    im1 = axs[0].imshow(co_sig, cmap='coolwarm', interpolation='none', origin='lower',vmin=min,vmax=max, extent=extent, aspect='auto') 
    axs[0].set_title('sWeights Signal')
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(co_DR, cmap='coolwarm', interpolation='none', origin='lower',vmin=min,vmax=max, extent=extent, aspect='auto') #, vmin=min, vmax=max , norm=SymLogNorm(linthresh=1, linscale=0.5,vmin=min, vmax=max),
    axs[1].set_title('Density Ratio Signal')
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    for ax in axs:
      ax.set_xlabel(r'$\phi$ [rad]')
      ax.set_ylabel(r'$\phi$ [rad]')
      ax.set_xticks(tick_positions)
      ax.set_xticklabels(tick_labels) #, rotation=90)
      ax.set_yticks(tick_positions)
      ax.set_yticklabels(tick_labels)

    fig.suptitle('Bin-to-Bin Covariance Matrices')
    plt.tight_layout()
    plt.savefig(self.print_dir+'BinsCov_Comp_phi'+self.endName+'.png')

  def plotPulls(self,h,hDR,herr,herrDR,edges,edgesDR):

    std = np.sqrt(herr**2 + herrDR**2)              # shape (N, n_bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        pulls = np.where((std != 0),
                         (h - hDR) / std,
                         0.0)                           # shape (N, n_bins)
        pulls_err = np.where((std != 0),
                             1.0,
                             0.0)                       # shape (N, n_bins)

    #one D example
    #std=np.sqrt(errsig[i]**2 + errpred[i]**2) #sum of squares of sqrt(nsig) and sqrt(npred)
    #res.append((nsig[i]-npred[i])/std)

    # Bin centers (same for all histograms)
    plotlocs = 0.5 * (edges[0,1:] + edges[0,:-1])            # shape (n_bins,)

    print(plotlocs.shape)
    print(h.shape)

    fig = plt.figure(figsize=(20, 20))
    plt.hist(pulls.flatten(), range=(-2,2),bins=100,color='royalblue',label='sWeighted Signal')
    plt.xlabel('$g_{hist}$')
    plt.title(r'$g_{hist}$ in $\phi$')
    plt.savefig(self.print_dir+'PullsAllIts'+self.endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    plt.hist(pulls[0], range=(-2,2),bins=100,color='royalblue',label='sWeighted Signal')
    plt.xlabel('$g_{hist}$')
    plt.title(r'$g_{hist}$ in $\phi$')
    plt.savefig(self.print_dir+'PullsFirstIt'+self.endName+'.png')

    # Per-histogram (row-wise) mean and std of pulls
    pull_means = np.mean(pulls, axis=1)         # shape (N,)
    pull_stds = np.std(pulls, axis=1, ddof=1)   # shape (N,)

    std_means = np.mean(std, axis=1)         # shape (N,)
    std_stds = np.std(std, axis=1, ddof=1)   # shape (N,)

    # Global statistics on pull_means and pull_stds
    mean_of_means = np.mean(pull_means)
    std_of_means = np.std(pull_means, ddof=1)

    mean_of_stds = np.mean(pull_stds)
    std_of_stds = np.std(pull_stds, ddof=1)

    mean_of_err = np.mean(std_means)
    std_of_err = np.std(std_means, ddof=1)

    mean_of_err_std = np.mean(std_stds)
    std_of_err_std = np.std(std_stds, ddof=1)

    print('\nMean and Std of all pull means:')
    print(mean_of_means, std_of_means)

    print('\nMean and Std of all pull std:')
    print(mean_of_stds, std_of_stds)


    print('\nMean and Std of all sqrt(2)*err means:')
    print(mean_of_err, std_of_err)

    print('\nMean and Std of all sqrt(2)*err stds:')
    print(mean_of_err_std, std_of_err_std)

    # mean_of_vals = np.mean(h)
    # std_of_vals = np.std(h, ddof=1)

    # mean_of_histoerr = np.mean(herr)
    # std_of_histoerr = np.std(herr, ddof=1)

    # mean_of_difs = np.mean(h-hDR)
    # std_of_difs = np.std(h-hDR, ddof=1)

    # print('\nMean and Std of all histo values:')
    # print(mean_of_vals, std_of_vals)

    # print('\nMean and Std of all histo differences:')
    # print(mean_of_difs, std_of_difs)

    # print('\nMean and Std of all err:')
    # print(mean_of_histoerr, std_of_histoerr)

    fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True, sharey=False,tight_layout=True,height_ratios=[2, 1])
    histo = axs[0].errorbar(plotlocs, h[0], herr[0], fmt='.',color='royalblue', label="sWeights Signal",linewidth=3,markersize=25,capsize=7,elinewidth=5)
    histo = axs[0].errorbar(plotlocs, hDR[0], herrDR[0], fmt='.',color='firebrick', label="Density Ratio Signal",linewidth=3,markersize=25,capsize=7,elinewidth=5)
    axs[0].legend(loc='upper right')
    ymin, ymax = axs[0].get_ylim()
    if ymin<0:
      axs[0].set_ylim(ymin, ymax * 1.35)
    else:
      axs[0].set_ylim(0, ymax * 1.35)
    axs[0].set_title(r'$\phi$')
    axs[0].axhline(y=0, color='grey', linestyle='--')
        
    axs[1].errorbar(x=plotlocs, y=pulls[0], yerr=pulls_err[0],fmt='s', color='mediumorchid',ms=10,capsize=7,elinewidth=3)
    axs[1].axhline(y = 0.0, color = 'black', linestyle = '--')#,label='1') 
    axs[1].axhline(y = -1.0, color = 'grey', linestyle = '--')#,label='1') 
    axs[1].axhline(y = 1.0, color = 'grey', linestyle = '--')#,label='1') 
    axs[1].axhline(y = -2.0, color = 'silver', linestyle = '--')#,label='1') 
    axs[1].axhline(y = 2.0, color = 'silver', linestyle = '--')#,label='1') 
    axs[1].set_ylim(-5, 5)
    axs[1].set_ylabel('Pull')
    plt.xlabel(r'$\phi$ [rad]')
    plt.savefig(self.print_dir+'PullsFirstItHistoComp'+self.endName+'.png')

    
      
