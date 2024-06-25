import ROOT
import numpy as np
import time as timeCount
import math
import random 
import zfit
import mplhep
from matplotlib import pyplot as plt
from hep_ml import splot
import pandas

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

class generator:

  masses=np.array([0.,0.])
  targetMass=0.1349768
  nSignalEvents=1000
  nBGEvents=1000

  signalData=np.zeros((1000,3*2))
  bgData=np.zeros((1000,3*2))
  mixedData=np.zeros((1000,3*2))

  signalIM=np.zeros((1,1))
  signalBG=np.zeros((1,1))
  mixedIM=np.zeros((1,1))

  sPlotModel=0
  sPlotData=np.zeros((1,1))
  sWeights=np.zeros((1,1))



  def __init__(self,ms,t,nEvSig,nEvBG):
    self.masses=ms
    self.targetMass=t
    self.nSignalEvents=nEvSig
    self.nBGEvents=nEvBG
    self.signalData=np.zeros((nEvSig,3*len(ms)))
    self.bgData=np.zeros((nEvBG,3*len(ms)))

    self.genSignal()
    self.genBackground()


  def genSignal(self):

    event = ROOT.TGenPhaseSpace()

    start_time = timeCount.time()

    for it in range(self.nSignalEvents):
      target = ROOT.TLorentzVector(0.0,0.0,0.0,self.targetMass)
      event.SetDecay(target,len(self.masses),self.masses,"")
      event.Generate()

      for j in range(len(self.masses)):
        self.signalData[it,j*3+0]=event.GetDecay(j).P()
        self.signalData[it,j*3+1]=event.GetDecay(j).Theta()
        self.signalData[it,j*3+2]=event.GetDecay(j).Phi()

      if (it%50000) == 0 and it!=0:
        fin_time = timeCount.time()
        tdif=fin_time-start_time
        print('Generated '+str(it)+' signal events out of '+str(self.nSignalEvents)+' in '+format(tdif,'.2f')+'s')

    fin_time = timeCount.time()
    tdif=fin_time-start_time 
    print('Generated all '+str(self.nSignalEvents)+' signal events in '+format(tdif,'.2f')+'s')

  def sample(self,IM):
    r=np.random.exponential(0.5)
    if(r>IM):
      return True
    else:
      return False

  def genBackground(self):

    
    event = ROOT.TGenPhaseSpace()

    start_time = timeCount.time()

    it=0
    while it<self.nBGEvents:
      mass = random.uniform(self.targetMass-0.3*self.targetMass,self.targetMass+0.3*self.targetMass)
      target = ROOT.TLorentzVector(0.0,0.0,0.0,mass)
      event.SetDecay(target,len(self.masses),self.masses,"")

      event.Generate()

      eve=np.zeros((1,3*len(self.masses)))
      for j in range(len(self.masses)):
        eve[0,j*3+0]=event.GetDecay(j).P()
        eve[0,j*3+1]=event.GetDecay(j).Theta()
        eve[0,j*3+2]=event.GetDecay(j).Phi()

      IM=self.calcInvariantMass(eve)[0]

      #shape bg with exponential 
      if self.sample(IM)==True:
        self.bgData[it]=eve
        it=it+1

        if (it%50000) == 0 and it!=0:
          fin_time = timeCount.time()
          tdif=fin_time-start_time
          print('Generated '+str(it)+' background events out of '+str(self.nBGEvents)+' in '+format(tdif,'.2f')+'s')
    
    fin_time = timeCount.time()
    tdif=fin_time-start_time 
    print('Generated all '+str(self.nBGEvents)+' background events in '+format(tdif,'.2f')+'s')

    self.mixedData=np.vstack((self.bgData,self.signalData))
    np.random.shuffle(self.mixedData)
    self.smear()
    self.setIMArrays()
    self.scale()
    


  def setIMArrays(self):
    
    self.mixedIM=self.calcInvariantMass(self.mixedData)
    self.bgIM=self.calcInvariantMass(self.bgData)
    self.signalIM=self.calcInvariantMass(self.signalData)
    
    mask_h=self.mixedIM<(self.targetMass+self.targetMass*0.25)
    mask_l=self.mixedIM>(self.targetMass-self.targetMass*0.25)
    self.mixedData=self.mixedData[mask_l & mask_h]
    self.mixedIM=self.mixedIM[mask_l & mask_h]

    mask_h=self.bgIM<(self.targetMass+self.targetMass*0.25)
    mask_l=self.bgIM>(self.targetMass-self.targetMass*0.25)
    self.bgData=self.bgData[mask_l & mask_h]
    self.bgIM=self.bgIM[mask_l & mask_h]

    mask_h=self.signalIM<(self.targetMass+self.targetMass*0.25)
    mask_l=self.signalIM>(self.targetMass-self.targetMass*0.25)
    self.signalData=self.signalData[mask_l & mask_h]
    self.signalIM=self.signalIM[mask_l & mask_h]

  def scale(self):
    for i in range(len(self.masses)):
      self.signalData[:,i*3+1]=(self.signalData[:,i*3+1])/3.5
      self.signalData[:,i*3+2]=(self.signalData[:,i*3+2] + 3.5)/7.0

      self.bgData[:,i*3+1]=(self.bgData[:,i*3+1])/3.5
      self.bgData[:,i*3+2]=(self.bgData[:,i*3+2] + 3.5)/7.0

      self.mixedData[:,i*3+1]=(self.mixedData[:,i*3+1])/3.5
      self.mixedData[:,i*3+2]=(self.mixedData[:,i*3+2] + 3.5)/7.0

  def calcInvariantMass(self,arr):

    sumE=0
    sumPx=0
    sumPy=0
    sumPz=0

    for i in range(len(self.masses)):
      sumPx = sumPx + arr[:,i*3+0]*np.sin(arr[:,i*3+1])*np.cos(arr[:,i*3+2])
      sumPy = sumPy + arr[:,i*3+0]*np.sin(arr[:,i*3+1])*np.sin(arr[:,i*3+2])
      sumPz = sumPz + arr[:,i*3+0]*np.cos(arr[:,i*3+1])
      sumE= sumE + np.sqrt(np.square(arr[:,i*3+0])+self.masses[i]*self.masses[i])
      
    IM = np.sqrt(np.square(sumE)- ( np.square(sumPx) + np.square(sumPy) + np.square(sumPz) ))
    
    return IM
  
  def getSignal(self):
    return self.signalData, self.signalIM
  
  def getBackground(self):
    return self.bgData, self.bgIM
  
  def getMixedData(self):
    return self.mixedData, self.mixedIM
  
  def smear(self):
        
    rs=np.random.uniform(-0.025,0.025,self.signalData.shape)
    self.signalData=self.signalData+rs

    rbg=np.random.uniform(-0.025,0.025,self.bgData.shape)
    self.bgData=self.bgData+rbg
    
    self.mixedData=np.vstack((self.bgData,self.signalData))
    np.random.shuffle(self.mixedData)

  def computesWeights(self):
    
    #range over which to fit
    obs = zfit.Space('mass', (self.targetMass-0.25*self.targetMass, self.targetMass+0.25*self.targetMass))

    #gaussian pdf for signal
    mu = zfit.Parameter('mu', self.targetMass, self.targetMass-0.1*self.targetMass, self.targetMass+0.1*self.targetMass)
    sigma = zfit.Parameter('sigma',0.025, 0., 10)
    signal_pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    #exponential pdf for background
    lambd = zfit.Parameter('lambda', -5, -10, 10)
    comb_bkg_pdf = zfit.pdf.Exponential(lambd, obs=obs)

    #range for species yield
    sig_yield = zfit.Parameter('sig_yield', self.nSignalEvents, 0,  self.nSignalEvents*10,
                                step_size=1)  # step size: default is small, use appropriate
    bkg_yield = zfit.Parameter('bkg_yield', self.nBGEvents, 0, self.nBGEvents, step_size=1)

    # Create the extended models
    extended_sig = signal_pdf.create_extended(sig_yield)
    extended_bkg = comb_bkg_pdf.create_extended(bkg_yield)

    # The final model is the combination of the signal and backgrond PDF
    self.sPlotModel = zfit.pdf.SumPDF([extended_bkg, extended_sig])

    # Builds the loss.
    self.sPlotData = zfit.Data.from_numpy(obs=obs, array=self.mixedIM)
    nll_sw = zfit.loss.ExtendedUnbinnedNLL(self.sPlotModel, self.sPlotData)

    # Minimizes the loss.
    minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
    result_sw = minimizer.minimize(nll_sw)
    print(result_sw.params)

    zfit.param.set_values(nll_sw.get_params(), result_sw)

    #Use hep-ml to comput weights
    probs = pandas.DataFrame(dict(sig=self.sPlotModel.get_models()[1].ext_pdf(self.sPlotData), bck=self.sPlotModel.get_models()[0].ext_pdf(self.sPlotData)))
    probs = probs.div(probs.sum(axis=1), axis=0)  

    sWeights = splot.compute_sweights(probs)
    
    return sWeights


