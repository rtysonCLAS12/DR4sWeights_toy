from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import mplhep
import numpy as np

class plotter:
  print_dir=''
  endName=''
  targetMass=0.1349768

  def __init__(self,t,pd='',en=''):
    self.targetMass=t
    self.print_dir=pd
    self.endName=en

  def plotIMComparison(self,IMSig,IMBG,IMAll):
    fig = plt.figure(figsize=(20, 20))
    plt.hist(IMSig, range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,color='royalblue',label='Generated Signal')
    plt.hist(IMBG, range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,edgecolor='firebrick',label='Generated Background',hatch='/', histtype='step',fill=False,linewidth=3)
    plt.hist(IMAll, range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
    plt.legend(loc='upper right')
    plt.xlabel("Invariant Mass [GeV]")
    plt.title("Invariant Mass")
    plt.savefig(self.print_dir+'genIM'+self.endName+'.png')

  def plotSWeightedVariables(self,vars,sWeights,sWeightsBG,nPart):

    names=['_P','_Theta','_Phi']
    titles=['P',r'$\theta$',r'$\phi$']
    units=['[GeV]','[rad]','[rad]']
    #lims_l=[0,0,-3.5]
    #lims_h=[1,3.5,3.5]
    lims_l=[0,0,0]
    lims_h=[1,1,1]


    for i in range(nPart):
      for j in range(3):

        pName='_part'+str(i)
        pTitle='Particle '+str(i)

        fig = plt.figure(figsize=(20, 20))
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,color='royalblue',label='sWeights Signal',weights=sWeights)
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,edgecolor='firebrick',label='sWeights Background',hatch='/', histtype='step',fill=False,linewidth=3,weights=sWeightsBG)
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
        plt.legend(loc='upper right')
        plt.xlabel(pTitle+' '+titles[j]+' '+units[j])
        plt.title(pTitle+' '+titles[j])
        plt.savefig(self.print_dir+'sWeights'+pName+names[j]+self.endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    plt.hist(vars[:,-1], range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,color='royalblue',label='sWeights Signal',weights=sWeights)
    plt.hist(vars[:,-1], range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,edgecolor='firebrick',label='sWeights Background',hatch='/', histtype='step',fill=False,linewidth=3,weights=sWeightsBG)
    plt.hist(vars[:,-1], range=[self.targetMass-0.25*self.targetMass,self.targetMass+0.25*self.targetMass],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
    plt.legend(loc='upper right')
    plt.xlabel("Invariant Mass [GeV]")
    plt.title("Invariant Mass")
    plt.savefig(self.print_dir+'sWeights_IM'+self.endName+'.png')

  def plotDRToSWeightComp(self,vars,sWeights,DRWeights,nPart):

    #names=['_Theta','_Phi']
    #titles=[r'$\theta$',r'$\phi$']
    #units=['[rad]','[rad]']
    #lims_l=[0,0]
    #lims_h=[1,1]

    names=['_P','_Theta','_Phi']
    titles=['P',r'$\theta$',r'$\phi$']
    units=['[GeV]','[rad]','[rad]']
    #lims_l=[0,0,-3.5]
    #lims_h=[1,3.5,3.5]
    lims_l=[0,0,0]
    lims_h=[1,1,1]

    for i in range(nPart):
      for j in range(3): #2

        pName='_part'+str(i)
        pTitle='Particle '+str(i)

        fig = plt.figure(figsize=(20, 20))
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,color='royalblue',label='sWeights Signal',weights=sWeights) #*2
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,edgecolor='firebrick',label='Density Ratio Signal',hatch='/', histtype='step',fill=False,linewidth=3,weights=DRWeights)
        plt.hist(vars[:,i*3+j], range=[lims_l[j],lims_h[j]],bins=100,edgecolor='black',color='black',label='All', histtype='step',fill=False,linewidth=3)
        plt.legend(loc='upper right')
        plt.xlabel(pTitle+' '+titles[j]+' '+units[j])
        plt.title(pTitle+' '+titles[j])
        plt.savefig(self.print_dir+'DRsWeightsComp'+pName+names[j]+self.endName+'.png')


  def plot_fit_projection(self,model, data, nbins=30, ax=None):

    fig = plt.figure(figsize=(20, 20))
    # The function will be reused.
    if ax is None:
      ax = plt.gca()

    lower, upper = data.data_range.limit1d

    # Creates and histogram of the data and plots it with mplhep.
    counts, bin_edges = np.histogram(data.unstack_x(), bins=nbins)
    mplhep.histplot(counts, bins=bin_edges, histtype="errorbar", yerr=True,
                    label="Data", ax=ax, color="black",markersize=25)

    binwidth = np.diff(bin_edges)[0]
    x = np.linspace(lower, upper, num=1000)  # or np.linspace

    # Line plots of the total pdf and the sub-pdfs.
    y = model.ext_pdf(x) * binwidth
    for m, l, c in zip(model.get_models(), ["background", "signal"], ["firebrick", "royalblue"]):
        ym = m.ext_pdf(x) * binwidth
        ax.plot(x, ym, label=l, color=c,linewidth=4)
    ax.plot(x, y, label="total", color="mediumorchid",linewidth=4)

    ax.set_title("Fitted Invariant Mass")
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Invariant Mass [GeV]")
    ax.legend()
    plt.savefig(self.print_dir+'fittedIM'+self.endName+'.png')

    return ax
