from generator import generator
from trainer import trainer

import numpy as np
from iminuit import Minuit, cost
from iminuit.cost import ExtendedUnbinnedNLL
import math
import time

class performance:

  trer=None
  verbose=True

  def __init__(self,tr,verbose=True):
    self.trer=tr
    self.verbose=verbose

  def fitAsymmetryDRW(self,gen,dataIn,drWeightsIn,swWeightsIn):
    #print(drWeightsIn)
    #print(swWeightsIn)
    
    mass_dist=dataIn[:,0]
    sigFit,bgFit,yieldsFit=gen.mass_splot_fit(mass_dist)
    phi_dist = dataIn[:,1]
    phibins = np.linspace(gen.Phmin, gen.Phmax, 100)

    sig_sumweights, edges = np.histogram( phi_dist, weights=drWeightsIn, bins=phibins )
    sig_sumweight_sqrd, edges = np.histogram( phi_dist, weights=( swWeightsIn*swWeightsIn ), bins=phibins )
    errors = np.sqrt(sig_sumweight_sqrd)
    centres = (edges[:-1] + edges[1:]) / 2
    
    
    c = cost.LeastSquares(centres, sig_sumweights, errors, gen.AsymmetryN)
    m1 = Minuit(c, Sigma=0.1, N=yieldsFit[0]/edges.size )
    m1.migrad()
    
    if self.verbose==True:
      print('\n DR Asymmetry Fit Results: ')
      #print(m1)
      print('Sigma='+format(m1.values[0],'.4f')+' +/- '+format(m1.errors[0],'.4f'))
      print('N='+format(m1.values[1],'.0f')+' +/- '+format(m1.errors[1],'.0f'))
      print('chi2/N='+format(m1.fval/(phibins.size),'.4f') )
      print('fit pull mean '+format(np.mean(c.pulls(m1.values)),'.4f'))
      print('fit pull std '+format(np.std(c.pulls(m1.values)),'.4f')+'\n\n') 

    return m1.values[0],m1.errors[0],m1.values[1],m1.errors[1],np.mean(c.pulls(m1.values)),np.std(c.pulls(m1.values)),m1.fval/(phibins.size)
        
    
  def fitAsymmetrySPlot(self,gen,dataIn,sigWeightsIn) :
      
    mass_dist=dataIn[:,0]
    sigFit,bgFit,yieldsFit=gen.mass_splot_fit(mass_dist)
    phi_dist = dataIn[:,1]
    phibins = np.linspace(gen.Phmin, gen.Phmax, 100)

    sig_sumweights, edges = np.histogram( phi_dist, weights=sigWeightsIn, bins=phibins )
    sig_sumweight_sqrd, edges = np.histogram( phi_dist, weights=( sigWeightsIn*sigWeightsIn ), bins=phibins )
    errors = np.sqrt(sig_sumweight_sqrd)
    centres = (edges[:-1] + edges[1:]) / 2
    
    
    c = cost.LeastSquares(centres, sig_sumweights, errors, gen.AsymmetryN)
    m1 = Minuit(c, Sigma=0.1, N=yieldsFit[0]/edges.size )
    m1.migrad()
    
    if self.verbose==True:
      print('\n sPlot Asymmetry Fit Results: ')
      #print(m1)
      print('Sigma='+format(m1.values[0],'.4f')+' +/- '+format(m1.errors[0],'.4f'))
      print('N='+format(m1.values[1],'.0f')+' +/- '+format(m1.errors[1],'.0f'))
      print('chi2/N='+format(m1.fval/(phibins.size),'.4f') )
      print('fit pull mean '+format(np.mean(c.pulls(m1.values)),'.4f'))
      print('fit pull std '+format(np.std(c.pulls(m1.values)),'.4f')+'\n\n') 

    return m1.values[0],m1.errors[0],m1.values[1],m1.errors[1],np.mean(c.pulls(m1.values)),np.std(c.pulls(m1.values)),m1.fval/(phibins.size)
  
  def bootstrap_sample(self,all_data) :
    nrows,ncols = all_data.shape
    Nsamp = nrows
    #choose to sample the array indices
    all_indices = np.arange(Nsamp)
    #samle Nk events
    indices_bt = np.random.choice(all_indices,Nsamp)
    return all_data[indices_bt]

    
  def bootstrap_sample_synched(self,all_data, all_weights) :
    nrows,ncols = all_data.shape
    Nsamp = nrows
    #now we need to synchronise our bootstrap samples 
    #so we instead choose to sample the array indices
    all_indices = np.arange(Nsamp)
    #samle Nk events
    indices_bt = np.random.choice(all_indices,Nsamp)
    return all_data[indices_bt],all_weights[indices_bt]


  def do_bootstrap_fits(self,nboot,gen,dataIn,sigWeightsIn):
    #given data and weights perform nboot bootstrapped fits
    sigma = np.zeros((nboot,1))
    sigma_err = np.zeros((nboot,1))
    N = np.zeros((nboot,1))
    N_err = np.zeros((nboot,1))
    pull_mean = np.zeros((nboot,1))
    pull_std = np.zeros((nboot,1))
    chi2 = np.zeros((nboot,1))
     
    startT_it = time.time()
    for iboot in range(0,nboot):
      endT_it = time.time()
      T_it=(endT_it-startT_it)

      if iboot==0:
        print('performing fit bootstrap :',iboot)
      else:
        print('performing fit bootstrap :'+str(iboot)+' time since start '+format(T_it,'.2f')+'s')
      obs, weights = self.bootstrap_sample_synched(dataIn,sigWeightsIn)
      sigma[iboot,0],sigma_err[iboot,0],N[iboot,0],N_err[iboot,0],pull_mean[iboot,0],pull_std[iboot,0],chi2[iboot,0] = self.fitAsymmetryDRW(gen,obs,weights)
      #sigma[iboot,0]  = self.fitAsymmetryDRW(gen,obs,weights)

    if self.verbose==True:
      print('\n\n *****Mean and std of fit bootstrapping*****')
      print('sigma = ',np.mean(sigma),'sigma std = ',np.std(sigma),'sigma error =',np.mean(sigma_err))
      print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
      print('pull std ',np.mean(pull_std),np.std(pull_std))

    return np.hstack((sigma, sigma_err,pull_mean,pull_std,chi2))

  def do_bootstrap_splot(self,nboot,gen) :
    #given discriminatory variable xdisc,
    #generate bootstrap sample and perform splot fit on it
    sigma = np.zeros((nboot,1))
    sigma_err = np.zeros((nboot,1))
    N = np.zeros((nboot,1))
    N_err = np.zeros((nboot,1))
    pull_mean = np.zeros((nboot,1))
    pull_std = np.zeros((nboot,1))
    chi2 = np.zeros((nboot,1))
     
    startT_it = time.time()
    for iboot in range(0,nboot):
      endT_it = time.time()
      T_it=(endT_it-startT_it)
      
      if iboot==0:
        print('performing splot bootstrap :',iboot)
      else:
        print('performing splot bootstrap :'+str(iboot)+' time since start '+format(T_it,'.2f')+'s')
      #print('do_bootstrap_splot',gen.getData().shape)
      boot_data = self.bootstrap_sample(gen.getData())
      #print('boot_data',boot_data.shape)
      boot_weights,bck_weights = gen.computesWeights(boot_data[:,0])
      sigma[iboot,0],sigma_err[iboot,0],N[iboot,0],N_err[iboot,0],pull_mean[iboot,0],pull_std[iboot,0],chi2[iboot,0] = self.fitAsymmetrySPlot(gen,boot_data,boot_weights)
      
    if self.verbose==True:
      print('\n\n *****Mean and std of sPlot bootstrapping*****')
      print('sigma = ',np.mean(sigma),'sigma std = ',np.std(sigma),'sigma error =',np.mean(sigma_err))
      print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
      print('pull std ',np.mean(pull_std),np.std(pull_std))

    return np.hstack((sigma, sigma_err,pull_mean,pull_std,chi2))

  def train(self,gen,all_data,all_weights) :
    all_data=gen.scale(all_data)
    
    self.trer.clear_models()
    self.trer.train(all_data,all_weights,verbose=False)
    weights_DR=self.trer.predict(all_data,verbose=False)
    
    all_data=gen.unscale(all_data)
    
    return weights_DR

  def do_bootstrap_dr(self,nboot,gen) :
    #given discriminatory variable xdisc,
    #generate bootstrap sample and perform splot fit on it
    sigma = np.zeros((nboot,1))
    sigma_err = np.zeros((nboot,1))
    N = np.zeros((nboot,1))
    N_err = np.zeros((nboot,1))
    pull_mean = np.zeros((nboot,1))
    pull_std = np.zeros((nboot,1))
    chi2 = np.zeros((nboot,1))
    bsigma = np.zeros((nboot,1))
    bsigma_err = np.zeros((nboot,1))
    bN = np.zeros((nboot,1))
    bN_err = np.zeros((nboot,1))
    bpull_mean = np.zeros((nboot,1))
    bpull_std = np.zeros((nboot,1))
    bgchi2 = np.zeros((nboot,1))

    startT_it = time.time()
    for iboot in range(0,nboot):
      endT_it = time.time()
      T_it=(endT_it-startT_it)
      
      if iboot==0:
        print('performing DR bootstrap :',iboot)
      else:
        print('performing DR bootstrap :'+str(iboot)+' time since start '+format(T_it,'.2f')+'s')
      #print('do_bootstrap_splot',gen.getData())
      boot_data = self.bootstrap_sample(gen.getData())
      #print('boot_data',boot_data)
      boot_weights,bck_weights = gen.computesWeights(boot_data[:,0])
      boot_weights=np.asarray(boot_weights).reshape((boot_data.shape[0],1))
      dr_weights = self.train(gen,boot_data,boot_weights)
      sigma[iboot,0],sigma_err[iboot,0],N[iboot,0],N_err[iboot,0],pull_mean[iboot,0],pull_std[iboot,0],chi2[iboot,0] = self.fitAsymmetryDRW(gen,boot_data,dr_weights,boot_weights[:,0].T)
      
      #print('now try background weights')
      #bck_weights=np.asarray(bck_weights).reshape((boot_data.shape[0],1))
      #dr_weights = self.train(gen,boot_data,bck_weights)
      #dr_weights= 1 - dr_weights
      #bsigma[iboot,0],bsigma_err[iboot,0],bN[iboot,0],bN_err[iboot,0],bpull_mean[iboot,0],bpull_std[iboot,0],bgchi2[iboot,0] = self.fitAsymmetryDRW(gen,boot_data,dr_weights,boot_weights[:,0].T)

    if self.verbose==True:
      print('\n\n *****Mean and std of DR bootstrapping*****')
      print('sigma = ',np.mean(sigma),'sigma std = ',np.std(sigma),'sigma error =',np.mean(sigma_err))
      print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
      print('pull std ',np.mean(pull_std),np.std(pull_std))
      #print('bg sigma ',np.mean(bsigma),np.std(bsigma),np.mean(bsigma_err))
      #print('bg pull mean ',np.mean(bpull_mean),np.std(bpull_mean))
      #print('bg pull std ',np.mean(bpull_std),np.std(bpull_std))

    return np.hstack((sigma, sigma_err,pull_mean,pull_std,chi2))
  
  def do_loop(self,nIt,gen):

    sigma = np.zeros((nIt,1))
    sigma_err = np.zeros((nIt,1))
    N = np.zeros((nIt,1))
    N_err = np.zeros((nIt,1))
    pull_mean = np.zeros((nIt,1))
    pull_std = np.zeros((nIt,1))
    chi2 = np.zeros((nIt,1))

    sigmaDR = np.zeros((nIt,1))
    sigmaDR_err = np.zeros((nIt,1))
    NDR = np.zeros((nIt,1))
    NDR_err = np.zeros((nIt,1))
    pullDR_mean = np.zeros((nIt,1))
    pullDR_std = np.zeros((nIt,1))
    chi2DR = np.zeros((nIt,1))

    startT_it = time.time()
    for it in range(nIt):

      endT_it = time.time()
      T_it=(endT_it-startT_it)

      if it==0:
        print('performing loop iteration :',it)
      else:
        print('performing loop iteration :'+str(it)+' time since start '+format(T_it,'.2f')+'s')

      gen.generate()
      all_data=gen.getData()
      all_weights,bck_weights = gen.computesWeights(all_data[:,0])

      all_data=gen.scale(all_data)
    
      self.trer.clear_models()
      self.trer.train(all_data,all_weights,verbose=False)
      dr_weights=self.trer.predict(all_data,verbose=False)

      all_data=gen.unscale(all_data)

      sigma[it,0],sigma_err[it,0],N[it,0],N_err[it,0],pull_mean[it,0],pull_std[it,0],chi2[it,0] = self.fitAsymmetrySPlot(gen,all_data,all_weights)
      sigmaDR[it,0],sigmaDR_err[it,0],NDR[it,0],NDR_err[it,0],pullDR_mean[it,0],pullDR_std[it,0],chi2DR[it,0] = self.fitAsymmetryDRW(gen,all_data,dr_weights,all_weights)

    if self.verbose==True:
      print('\n\n ***** Mean and std of sPlot Loop *****')
      print('sigma = ',np.mean(sigma),'sigma std = ',np.std(sigma),'sigma error =',np.mean(sigma_err))
      print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
      print('pull std ',np.mean(pull_std),np.std(pull_std))

      print('\n***** Mean and std of DR Loop *****')
      print('sigma = ',np.mean(sigmaDR),'sigma std = ',np.std(sigmaDR),'sigma error =',np.mean(sigmaDR_err))
      print('pull mean ',np.mean(pullDR_mean),np.std(pullDR_mean))
      print('pull std ',np.mean(pullDR_std),np.std(pullDR_std))

    return np.hstack((sigma, sigma_err,pull_mean,pull_std,chi2)), np.hstack((sigmaDR, sigmaDR_err,pullDR_mean,pullDR_std,chi2DR))
  
  def print_summary(self,perf,algoName,testName):

    print('\n\n ***** Mean and std of '+algoName+' '+testName+' *****')
    print('sigma = ',np.mean(perf[:,0]),'sigma std = ',np.std(perf[:,0]),'sigma error =',np.mean(perf[:,1]))
    print('pull mean ',np.mean(perf[:,2]),np.std(perf[:,2]))
    print('pull std ',np.mean(perf[:,3]),np.std(perf[:,3]))
    print('chi^2 ',np.mean(perf[:,4]),np.std(perf[:,4]))

  def write_out_summary(self, perf,algoName,testName,fName,opt):
    f = open(fName,opt)
    f.write('\n\n\n ***** Mean and std of '+algoName+' '+testName+' *****')
    f.write('\nsigma = '+str(np.mean(perf[:,0]))+'sigma std = '+str(np.std(perf[:,0]))+'sigma error ='+str(np.mean(perf[:,1])))
    f.write('\npull mean '+str(np.mean(perf[:,2]))+' '+str(np.std(perf[:,2])))
    f.write('\npull std '+str(np.mean(perf[:,3]))+' '+str(np.std(perf[:,3])))
    f.write('\nchi^2 '+str(np.mean(perf[:,4]))+' '+str(np.std(perf[:,4])))
    f.close()

      


