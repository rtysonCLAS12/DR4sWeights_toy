import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import optimize
from sweights import SWeight
import time as timeCount

from iminuit import Minuit, cost
from iminuit.cost import ExtendedUnbinnedNLL


class generator:

  Mmin = 0
  Mmax = 10
  Phmin = -np.pi
  Phmax = np.pi
  Zmin=-1
  Zmax=1

  nEvents=1000

  Data=np.zeros((1000,3*2))

  sigWeights=[]
  bgWeights=[]

  sigFit=[]
  bgFit=[]
  yieldsFit=[]


  def __init__(self,MRange,PhRange,ZRange,nEvs,generate=True):
    self.Mmin=MRange[0]
    self.Mmax=MRange[1]
    self.Phmin=PhRange[0]
    self.Phmax=PhRange[1]
    self.Zmin=ZRange[0]
    self.Zmax=ZRange[1]

    self.nEvents=nEvs

    if generate==True:
      self.generate()

  def getData(self):
    return self.Data

  #Unormalised Asymmetry
  def AsymmetryN(self,xphi,Sigma,N):
    return N*self.AsymmetryPDF(xphi,Sigma)

  #Asymmetry PDF
  def AsymmetryPDF(self,xphi,Sigma):
    return (1 - Sigma*np.cos(2*xphi))/(self.Phmax-self.Phmin)
  
  def SignalMassPDF(self,xmass,mean,width):
    sig  = norm(mean,width)
    #integral of signal function using CDF
    normInt = np.diff( sig.cdf([self.Mmin,self.Mmax]) )/ (self.Mmax-self.Mmin)
    #normalised PDF
    return sig.pdf(xmass)/normInt

  def Cheb(self,x,coeffs):
    return np.polynomial.chebyshev.chebval(x,coeffs)

  def BackGPDF(self,x,coeffs):

    chebedges = np.arange(-1.0, 1.0, 1./1000)
    chebcentres = (chebedges[:-1] + chebedges[1:]) / 2

    #transform x to 0 [-1,1]
    x = -1 + 2*(x-self.Mmin)/(self.Mmax-self.Mmin)
    val  = self.Cheb(x,coeffs)
    #integral of function (approximate)
    integ = np.sum(self.Cheb(chebcentres,coeffs))/chebcentres.size
    #pdf value
    return val/integ
  
  def TruePDF(self,m,ph,z):
    return self.SignalMassPDF(m,5,0.5)*self.AsymmetryPDF(ph,0.8) + 2*self.BackGPDF(m,[0.6,0.2])*self.AsymmetryPDF(ph,-0.2)

  def generate_event(self,gen_max_val,nEvs):
    x = np.random.uniform(self.Mmin,self.Mmax,nEvs)
    y = np.random.uniform(self.Phmin,self.Phmax,nEvs)
    z = np.random.uniform(self.Zmin,self.Zmax,nEvs)
    val = self.TruePDF(x,y,z)
    #print(val,max_val)
    mask=val > np.random.uniform(0,gen_max_val,nEvs)
    return x[mask],y[mask],z[mask]

  def getGenMaxVal(self):
    gen_max_val = 0.
    for i in range(0,1000):
      x = np.random.uniform(self.Mmin,self.Mmax)
      y = np.random.uniform(self.Phmin,self.Phmax)
      z = np.random.uniform(self.Zmin,self.Zmax)
      val = self.TruePDF(x,y,z)
      if val>gen_max_val :
        gen_max_val=val
        
    #increase max by 10% to be sure    
    gen_max_val*=1.1
    return gen_max_val
  
  def generate(self):

    #print('Get Sampling Max Value...')
    gen_max_val=self.getGenMaxVal()
    #print('Done')

    self.Data = np.zeros((1,1))

    start_time = timeCount.time()

    while self.Data.shape[0]<self.nEvents:

      if self.Data.shape[0]!=1:
        fin_time = timeCount.time()
        tdif=fin_time-start_time
        print('Generated '+str(self.Data.shape[0])+' signal events out of '+str(self.nEvents)+' in '+format(tdif,'.2f')+'s')

      x,y,z=self.generate_event(gen_max_val,self.nEvents)

      if self.Data.shape[0]==1:
        self.Data=np.hstack((x.reshape((x.shape[0],1)),y.reshape((x.shape[0],1)),z.reshape((x.shape[0],1))))
      else:
        t=np.hstack((x.reshape((x.shape[0],1)),y.reshape((x.shape[0],1)),z.reshape((x.shape[0],1))))
        self.Data=np.vstack((self.Data,t))

    fin_time = timeCount.time()
    tdif=fin_time-start_time 

    self.Data=self.Data[0:self.nEvents,:]

    print('Generated all '+str(self.nEvents)+' events in '+format(tdif,'.2f')+'s')

  def CombinedMassNExt(self,xmass,smean,swidth,bc0,bc1,bc2,Ys,Yb):
    return ((Ys+Yb),Ys*self.SignalMassPDF(xmass,smean,swidth)+Yb*self.BackGPDF(xmass,[bc0,bc1,bc2]))
  
  def mass_splot_fit(self,mass_dist):
    Ndata = mass_dist.size
    mi = Minuit( ExtendedUnbinnedNLL(mass_dist, self.CombinedMassNExt), smean=5, swidth=0.5,bc0=0.6,bc1=0.2,bc2=0, Ys=Ndata/2,Yb=Ndata/2 )
    mi.limits['Yb'] = (0,Ndata*1.1)
    mi.limits['Ys'] = (0,Ndata*1.1)
    mi.limits['smean'] = (self.Mmin,self.Mmax)
    mi.limits['swidth'] = (0.01,self.Mmax-self.Mmin)
    mi.limits['bc0'] = (-1,1)
    mi.limits['bc1'] = (-1,1)
    mi.limits['bc2'] = (-1,1)

    #fix overall normalisation coefficeint to 1
    #mi.fixed['bc0'] = True
    #mi.fixed['bc0'] = True
    mi.fixed['bc2'] = True

    #do fitting
    mi.migrad()

    #save values
    sg_mean=mi.values[0]
    sg_width=mi.values[1]
    bg_c0=mi.values[2]
    bg_c1=mi.values[3]
    bg_c2=mi.values[4]
    Ysignal = mi.values[5]
    Yback = mi.values[6]

    #print(mi)
    
    return [sg_mean,sg_width],[bg_c0,bg_c1,bg_c2],[Ysignal,Yback]

  
  def computesWeights(self,mass_dist):
    #mass_dist = self.Data[:,0]

    #print('\n\niMinuit Fit')
    self.sigFit,self.bgFit,self.yieldsFit = self.mass_splot_fit(mass_dist)

    spdf = lambda m: self.SignalMassPDF(m,self.sigFit[0],self.sigFit[1])
    bpdf = lambda m: self.BackGPDF(m,self.bgFit)

    # make the sweighter
    mrange = (self.Mmin,self.Mmax)

    sweighter = SWeight( mass_dist, [spdf,bpdf], self.yieldsFit, (mrange,), method='summation', compnames=('sig','bkg'), verbose=True, checks=True)

    self.sigWeights  = sweighter.get_weight(0, mass_dist)
    self.bgWeights  = sweighter.get_weight(1, mass_dist)

    return self.sigWeights,self.bgWeights
  
  def fitAsymmetry(self,DataIn,sigWeightsIn,sqdWeightsForErrIn,verbose=True):

    mass_dist=DataIn[:,0]
    sigFit,bgFit,yieldsFit=self.mass_splot_fit(mass_dist)

    phi_dist = DataIn[:,1]
    phibins = np.linspace(self.Phmin, self.Phmax, 100)

    sig_sumweights, edges = np.histogram( phi_dist, weights=sigWeightsIn, bins=phibins )
    sig_sumweight_sqrd, edges2 = np.histogram( phi_dist, weights=sqdWeightsForErrIn, bins=phibins ) #sigWeightsIn*sigWeightsIn
    errors = np.sqrt(sig_sumweight_sqrd)
    centres = (edges[:-1] + edges[1:]) / 2

    c = cost.LeastSquares(centres, sig_sumweights, errors, self.AsymmetryN)
    m1 = Minuit(c, Sigma=0.1, N=yieldsFit[0]/edges.size)
    m1.migrad()

    if verbose==True:
      print('\nAsymmetry Fit Results: ')
      #print(m1)
      print('Sigma='+format(m1.values[0],'.4f')+' +/- '+format(m1.errors[0],'.4f'))
      print('N='+format(m1.values[1],'.0f')+' +/- '+format(m1.errors[1],'.0f'))
      print('chi2/N='+format(m1.fval/(phibins.size),'.4f') )
      print(np.mean(c.pulls(m1.values)))
      print(np.std(c.pulls(m1.values))) 
    
    chi2=m1.fval/(phibins.size)
    return m1.values,c.pulls(m1.values),chi2


    return m1.values,c.pulls(m1.values),chi2
    
  def scale(self,data):
    data[:,0]=(data[:,0] - self.Mmin)/(self.Mmax - self.Mmin)
    data[:,1]=(data[:,1] - self.Phmin)/(self.Phmax - self.Phmin)
    data[:,2]=(data[:,2] - self.Zmin)/(self.Zmax - self.Zmin)
    return data

  def unscale(self,data):
    data[:,0]=(data[:,0] * (self.Mmax - self.Mmin) ) + self.Mmin
    data[:,1]=(data[:,1] * (self.Phmax - self.Phmin) ) + self.Phmin
    data[:,2]=(data[:,2] * (self.Zmax - self.Zmin) ) + self.Zmin
    return data
     

