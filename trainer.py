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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Input, Dense
from tensorflow.keras import optimizers as opt 
import tensorflow as tf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.base import clone

from joblib import dump, load

from energyflow.archs import PFN

import copy

class trainer:

  base_models=[]
  models=[]
  using_neuralResampler=[]
  bg_weights=[]

  def __init__(self,bms):
    self.base_models=bms
    
    self.clear_models()

  def clear_models(self):
    self.models.clear()
    self.bg_weights.clear()
    self.using_neuralResampler.clear()

    for b in self.base_models:
      if b=='PFN':
        #always use same NR architecture for simplicity as they can't be copied -_-
        self.models.append(PFN(input_dim=2,Phi_sizes= (1000,500,250, 100, 128), F_sizes=(1000,500,250, 100, 100),output_dim=1,output_act='sigmoid',loss='binary_crossentropy',metrics='',summary=False))
        self.using_neuralResampler.append(2)
      elif b=='NN':
        #being lazy now
        model = Sequential()
        model.add(Input(shape=(2,)))
        model.add(Dense(1024, activation='relu')) 
        model.add(Dense(512, activation='relu')) 
        model.add(Dense(256, activation='relu')) 
        model.add(Dense(128, activation='relu')) 
        model.add(Dense(64, activation='relu')) 
        model.add(Dense(32, activation='relu')) 
        model.add(Dense(16, activation='relu')) 
        model.add(Dense(1, activation='sigmoid'))
        opti=opt.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opti)
        self.models.append(model)
        self.using_neuralResampler.append(1)
      else:
        self.models.append(clone(b))
        self.using_neuralResampler.append(0)
    
  
  def getModels(self):
    return self.models

  def getBaseModels(self):
    return self.base_models

  def create_sample(self,data,sigWeights,bgWeights):
    #print(data.shape)
    Xall=np.vstack((data,data))
    weights=np.vstack((sigWeights.reshape((data.shape[0],1)),bgWeights.reshape((data.shape[0],1)))).reshape((Xall.shape[0]))
    Yall=np.vstack((np.ones((data.shape[0],1)),np.zeros((data.shape[0],1)))).reshape((Xall.shape[0]))
    
    #shuffle in unison
    p = np.random.permutation(Xall.shape[0])
    Xall=Xall[p]
    Yall=Yall[p]
    weights=weights[p]

    return Xall, Yall, weights

  def train(self,data,sigweights,verbose=True):
    startT_trainAll = time.time()

    self.bg_weights.append(np.ones((data.shape[0],1)))
    for i in range(len(self.models)):
      
      X_train, y_train, weights_train=self.create_sample(data,sigweights,self.bg_weights[i])

      startT_train = time.time()

      #don't include mass at var 0 in fit
      if self.using_neuralResampler[i]==0:
        self.models[i].fit(X_train[:,1:],y_train,sample_weight=weights_train)
      elif self.using_neuralResampler[i]==1:
        history=self.models[i].fit(X_train[:,1:],y_train,epochs=50, verbose=verbose,sample_weight=weights_train,batch_size=1000)
      else:
        #PFN models assume N particles in dimension 2
        history=self.models[i].fit(X_train[:,1:].reshape((X_train.shape[0],1,2)),y_train,epochs=100, verbose=verbose,sample_weight=weights_train,batch_size=1000)

      endT_train = time.time()
      T_train=(endT_train-startT_train)/60

      if verbose==True:
        print('It '+str(i)+': training single model took '+format(T_train,'.2f')+' minutes')

      new_bgweight=self.predict_oneModel(data,self.models[i],verbose=verbose,useNR=self.using_neuralResampler[i],modelNb=i)*self.bg_weights[i]
      self.bg_weights.append(new_bgweight)

    endT_trainAll = time.time()
    T_trainAll=(endT_trainAll-startT_trainAll)/60

    if verbose==True:
      print('\nTraining all models  took '+format(T_trainAll,'.2f')+' minutes\n')

      
  def predict_oneModel(self,data,model,useNR=0,verbose=True,modelNb=0):
    y_pred=np.ones((1,1))

    startT_test = time.time()

    if useNR==0:
      y_pred=model.predict_proba(data[:,1:])[:,1]
      #print(y_pred)
      #print(y_pred.shape)
    elif useNR==1:
      y_pred=model.predict(data[:,1:],batch_size=1000,verbose=verbose).reshape((data.shape[0]))
    else:
      y_pred=model.predict(data[:,1:].reshape((data.shape[0],1,2)),batch_size=1000,verbose=verbose).reshape((data.shape[0]))

    endT_test = time.time()
    T_test=(endT_test-startT_test)

    if verbose==True:
      print('Single model prediction took '+format(T_test,'.2f')+' seconds')

    #capping
    # y_pred[y_pred==1]=1-0.01
    # weights_DR = y_pred/(1-y_pred)
    # weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
    # weights_DR[weights_DR>10]=10 #some weights blow up
      
    #scaling instead
    # if modelNb==0:
    #   #y_pred[y_pred>0.5]=0
    #   y_pred[y_pred>0.75]=0.75
    # else:
    #   #y_pred = 0.5 + 0.5*(y_pred-0.5)
    #   y_pred[y_pred>0.75]=0.75 # ie weights >3
    # weights_DR = y_pred/(1-y_pred)
      
    # #capping v2
    y_pred[y_pred==1]=1-0.0000001
    weights_DR = y_pred/(1-y_pred)
    weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
    if modelNb==0:
      weights_DR[weights_DR>1]=1 #some weights blow up
    else:
      weights_DR[weights_DR>1]=1 #some weights blow up

    weights_DR=weights_DR.reshape((data.shape[0],1))

    return weights_DR

    #wrong reweight weights
    # if modelNb==0:
    #   y_pred[y_pred==1]=1-0.01
    #   weights_DR = y_pred/(1-y_pred)
    #   weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
    #   weights_DR[weights_DR>10]=1 #some weights blow up
    #   weights_DR=weights_DR.reshape((data.shape[0],1))
    #   return weights_DR
    # else:
    #   y_pred=y_pred.reshape((data.shape[0],1))
    #   return y_pred
  
  def predict(self,data,verbose=True):

    weights_DR_all=np.ones((data.shape[0]))

    startT_testAll = time.time()

    for i in range(len(self.models)):

      startT_test = time.time()

      if self.using_neuralResampler[i]==0:
        y_pred=self.models[i].predict_proba(data[:,1:])[:,1]
        #print(y_pred)
      elif self.using_neuralResampler[i]==1:
        y_pred=self.models[i].predict(data[:,1:],batch_size=1000,verbose=verbose).reshape((data.shape[0]))
      else:
        y_pred=self.models[i].predict(data[:,1:].reshape((data.shape[0],1,2)),batch_size=1000,verbose=verbose).reshape((data.shape[0]))

      endT_test = time.time()
      T_test=(endT_test-startT_test)

      if verbose==True:
        print('Single model prediction took '+format(T_test,'.2f')+' s')

      #capping
      # y_pred[y_pred==1]=1-0.01
      # weights_DR = y_pred/(1-y_pred)
      # weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
      # weights_DR[weights_DR>10]=1 #some weights blow up
        
      #scaling instead
      # if i==0:
      #   #y_pred[y_pred>0.5]=0
      #   y_pred[y_pred>0.75]=0.75
      # else:
      #   #y_pred = 0.5 + 0.5*(y_pred-0.5)
      #   y_pred[y_pred>0.75]=0.75 # ie weights >3
      # weights_DR = y_pred/(1-y_pred)
        
      #capping v2
      y_pred[y_pred==1]=1-0.0000001
      weights_DR = y_pred/(1-y_pred)
      weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
      if i==0:
        weights_DR[weights_DR>1]=1 #some weights blow up
      else:
        weights_DR[weights_DR>1]=1 #some weights blow up


      weights_DR_all=weights_DR_all*weights_DR

      #wrong reweight weights
      # if i==0:
      #   y_pred[y_pred==1]=1-0.01
      #   weights_DR = y_pred/(1-y_pred)
      #   weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
      #   weights_DR[weights_DR>10]=1 #some weights blow up
      #   weights_DR_all=weights_DR_all*weights_DR
      # else:
      #   weights_DR_all=weights_DR_all*y_pred
        

    endT_testAll = time.time()
    T_testAll=(endT_testAll-startT_testAll)

    if verbose==True:
      print('all models prediction took '+format(T_testAll,'.2f')+' s\n')

    return weights_DR_all


