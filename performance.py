from generator import generator

import numpy as np
from iminuit import Minuit, cost
from iminuit.cost import ExtendedUnbinnedNLL
import math
import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

class performance:


    def fitAsymmetryDRW(self,gen,dataIn,drWeightsIn,swWeightsIn):
        print(drWeightsIn)
        print(swWeightsIn)
        mass_dist=dataIn[:,0]
        sigFit=gen.sigFit
        bgFit=gen.bgFit
        yieldsFit=gen.yieldsFit

        phi_dist = dataIn[:,1]
        phibins = np.linspace(gen.Phmin, gen.Phmax, 100)
    
        sig_sumweights, edges = np.histogram( phi_dist, weights=drWeightsIn, bins=phibins )
        sig_sumweight_sqrd, edges = np.histogram( phi_dist, weights=( swWeightsIn*swWeightsIn ), bins=phibins )
        errors = np.sqrt(sig_sumweight_sqrd)
        centres = (edges[:-1] + edges[1:]) / 2
        
        
        c = cost.LeastSquares(centres, sig_sumweights, errors, gen.AsymmetryN)
        m1 = Minuit(c, Sigma=0.1, N=yieldsFit[0]/edges.size )
        m1.migrad()
        
        print('\nDR4W Asymmetry Fit Results: ')
        #print(m1)
        print('Sigma='+format(m1.values[0],'.4f')+' +/- '+format(m1.errors[0],'.4f'))
        print('N='+format(m1.values[1],'.0f')+' +/- '+format(m1.errors[1],'.0f'))
        print('chi2/N='+format(m1.fval/(phibins.size),'.4f') )
        
        print(np.mean(c.pulls(m1.values)))
        print(np.std(c.pulls(m1.values)))

        return m1.values[0],m1.errors[0],m1.values[1],m1.errors[1],np.mean(c.pulls(m1.values)),np.std(c.pulls(m1.values))
        
    
    def fitAsymmetrySPlot(self,gen,dataIn,sigWeightsIn) :
        mass_dist=dataIn[:,0]
        sigFit=gen.sigFit
        bgFit=gen.bgFit
        yieldsFit=gen.yieldsFit

        phi_dist = dataIn[:,1]
        phibins = np.linspace(gen.Phmin, gen.Phmax, 100)
    
        sig_sumweights, edges = np.histogram( phi_dist, weights=sigWeightsIn, bins=phibins )
        sig_sumweight_sqrd, edges = np.histogram( phi_dist, weights=( sigWeightsIn*sigWeightsIn ), bins=phibins )
        errors = np.sqrt(sig_sumweight_sqrd)
        centres = (edges[:-1] + edges[1:]) / 2
        
        
        c = cost.LeastSquares(centres, sig_sumweights, errors, gen.AsymmetryN)
        m1 = Minuit(c, Sigma=0.1, N=yieldsFit[0]/edges.size )
        m1.migrad()
        
        print('\n sPlot Asymmetry Fit Results: ')
        #print(m1)
        print('Sigma='+format(m1.values[0],'.4f')+' +/- '+format(m1.errors[0],'.4f'))
        print('N='+format(m1.values[1],'.0f')+' +/- '+format(m1.errors[1],'.0f'))
        print('chi2/N='+format(m1.fval/(phibins.size),'.4f') )
        
        print(np.mean(c.pulls(m1.values)))
        print(np.std(c.pulls(m1.values)))

        return m1.values[0],m1.errors[0],m1.values[1],m1.errors[1],np.mean(c.pulls(m1.values)),np.std(c.pulls(m1.values))
        
    def bootstrap_sample(self,all_data) :
        nrows,ncols = all_data.shape
        Nsamp = nrows
        #choose to sample the array indices
        all_indices = np.arange(Nsamp)
        #samle Nk events
        indices_bt = np.random.choice(all_indices,Nsamp)
        print('bootstrap_sample',all_data)
        print('bootstrap_sample',all_data[indices_bt])
        return all_data[indices_bt]

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
        sigma = np.zeros(nboot)
        sigma_err = np.zeros(nboot)
        N = np.zeros(nboot)
        N_err = np.zeros(nboot)
        pull_mean = np.zeros(nboot)
        pull_std = np.zeros(nboot)
         
        for iboot in range(0,nboot):
            print('performing bootstrap :',iboot)
            obs, weights = self.bootstrap_sample_synched(dataIn,sigWeightsIn)
            sigma[iboot],sigma_err[iboot],N[iboot],N_err[iboot],pull_mean[iboot],pull_std[iboot] = self.fitAsymmetryDRW(gen,obs,weights)
           # sigma[iboot]  = self.fitAsymmetryDRW(gen,obs,weights)
    


        print('sigma ',np.mean(sigma),np.std(sigma),np.mean(sigma_err))
        print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
        print('pull std ',np.mean(pull_std),np.std(pull_std))

    def do_bootstrap_splot(self,nboot,gen) :
        #given discriminatory variable xdisc,
        #generate bootstrap sample and perform splot fit on it
        sigma = np.zeros(nboot)
        sigma_err = np.zeros(nboot)
        N = np.zeros(nboot)
        N_err = np.zeros(nboot)
        pull_mean = np.zeros(nboot)
        pull_std = np.zeros(nboot)
         
        for iboot in range(0,nboot):
            print('performing splot bootstrap :',iboot)
            #print('do_bootstrap_splot',gen.Data)
            boot_data = self.bootstrap_sample(gen.Data)
            #print('boot_data',boot_data)
            boot_weights,bck_weights = gen.computesWeights(boot_data[:,0])
            sigma[iboot],sigma_err[iboot],N[iboot],N_err[iboot],pull_mean[iboot],pull_std[iboot] = self.fitAsymmetrySPlot(gen,boot_data,boot_weights)

        print('sigma ',np.mean(sigma),np.std(sigma),np.mean(sigma_err))
        print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
        print('pull std ',np.mean(pull_std),np.std(pull_std))


    def trainGBDT(self,gen,all_data,all_weights) :
        all_data=gen.scale(all_data)

        #training data is composed of twice the data
        #weighted with sWeights and by one
        Xall=np.vstack((all_data,all_data))
        weights=np.vstack((all_weights,np.ones((all_data.shape[0],1)))).reshape((Xall.shape[0]))
        Yall=np.vstack((np.ones((all_data.shape[0],1)),np.zeros((all_data.shape[0],1)))).reshape((Xall.shape[0]))
        
        #shuffle in unison
        p = np.random.permutation(Xall.shape[0])
        Xall=Xall[p]
        Yall=Yall[p]
        weights=weights[p]
        
        #split into training and testing sets
        nTrain=math.ceil(0.7*(Xall.shape[0]))
        #nTrain=Xall.shape[0]
        X_train=Xall[:nTrain,:]
        X_test=Xall[nTrain:,:]
    
        y_train=Yall[:nTrain]
        y_test=Yall[nTrain:]
        
        weights_train=weights[:nTrain]
        weights_test=weights[nTrain:]
        
        #both GradientBoosting and HistGradientBoosting work well
        #HistGradientBoosting is much faster
        
        model = HistGradientBoostingClassifier(max_depth=10, max_features=0.9)
        #model = GradientBoostingClassifier(max_depth=10)
        
        print('Training with '+str(X_train.shape[0])+' events...')
        
        #train model
        startT_train = time.time()
        
        #don't include mass at var 0 in fit
        model.fit(X_train[:,1:],y_train,sample_weight=weights_train)
        endT_train = time.time()
        T_train=(endT_train-startT_train)/60
        
        print('Training took '+format(T_train,'.2f')+' minutes\n')
        
        print('Test with '+str(X_test.shape[0])+' events...')
        
        #get predictions for all data
        startT_test = time.time()
        
        y_pred=model.predict_proba(all_data[:,1:])[:,1]
        
        endT_test = time.time()
        T_test=(endT_test-startT_test)
        
        print('Testing took '+format(T_test,'.4f')+' seconds\n')
        #we can now calculate the Density Ratio estimated Weights
        print('ypred =1',y_pred[y_pred==1].shape)
        y_pred[y_pred==1]=1-0.0000001
        weights_DR = y_pred/(1-y_pred)
        weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
        print('weights >1',weights_DR[weights_DR>1])
        weights_DR[weights_DR>1]=1 #some weights blow up

        all_data=gen.unscale(all_data)
        
        return weights_DR

    def do_bootstrap_splot4dr(self,nboot,gen) :
        #given discriminatory variable xdisc,
        #generate bootstrap sample and perform splot fit on it
        sigma = np.zeros(nboot)
        sigma_err = np.zeros(nboot)
        N = np.zeros(nboot)
        N_err = np.zeros(nboot)
        pull_mean = np.zeros(nboot)
        pull_std = np.zeros(nboot)
        bsigma = np.zeros(nboot)
        bsigma_err = np.zeros(nboot)
        bN = np.zeros(nboot)
        bN_err = np.zeros(nboot)
        bpull_mean = np.zeros(nboot)
        bpull_std = np.zeros(nboot)
    
        for iboot in range(0,nboot):
            print('performing splot bootstrap :',iboot)
            #print('do_bootstrap_splot',gen.Data)
            boot_data = self.bootstrap_sample(gen.Data)
            #print('boot_data',boot_data)
            boot_weights,bck_weights = gen.computesWeights(boot_data[:,0])
            boot_weights=np.asarray(boot_weights).reshape((boot_data.shape[0],1))
            dr_weights = self.trainGBDT(gen,boot_data,boot_weights)
            sigma[iboot],sigma_err[iboot],N[iboot],N_err[iboot],pull_mean[iboot],pull_std[iboot] = self.fitAsymmetryDRW(gen,boot_data,dr_weights,boot_weights[:,0].T)
           # print('now try background weights')
           # bck_weights=np.asarray(bck_weights).reshape((boot_data.shape[0],1))
           # dr_weights = self.trainGBDT(gen,boot_data,bck_weights)
           # dr_weights= 1 - dr_weights
            #bsigma[iboot],bsigma_err[iboot],bN[iboot],bN_err[iboot],bpull_mean[iboot],bpull_std[iboot] = self.fitAsymmetryDRW(gen,boot_data,dr_weights,boot_weights[:,0].T)

        print('sigma ',np.mean(sigma),np.std(sigma),np.mean(sigma_err))
        print('pull mean ',np.mean(pull_mean),np.std(pull_mean))
        print('pull std ',np.mean(pull_std),np.std(pull_std))
        print('bg sigma ',np.mean(bsigma),np.std(bsigma),np.mean(bsigma_err))
        print('bg pull mean ',np.mean(bpull_mean),np.std(bpull_mean))
        print('bg pull std ',np.mean(bpull_std),np.std(bpull_std))

