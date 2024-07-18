from generator import generator
from plotter import plotter
from performance import performance

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

from sklearn.base import clone

from joblib import dump, load

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, MaxPool2D
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras import optimizers as opt 
from tensorflow.keras import metrics as mt
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model

class trainer:

  base_models=[]
  models=[]
  using_neuralResampler=False
  bg_weights=[]

  def __init__(self,bms,isNR=False):
    self.base_models=bms
    self.using_neuralResampler=isNR
    if self.using_neuralResampler==False:
      for b in self.base_models:
        self.models.append(clone(b))
    else:
      for b in self.base_models:
        self.models.append(clone_model(b))
  
  def getModels(self):
    return self.models

  def getBaseModels(self):
    return self.base_models
  
  def clear_models(self):
    self.models.clear()
    if self.using_neuralResampler==False:
      for b in self.base_models:
        self.models.append(clone(b))
    else:
      for b in self.base_models:
        self.models.append(clone_model(b))
    self.bg_weights.clear()

  def create_sample(self,data,sigWeights,bgWeights):
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
      if self.using_neuralResampler==False:
        self.models[i].fit(X_train[:,1:],y_train,sample_weight=weights_train)
      else:
        history=self.models[i].fit(X_train[:,1:],y_train,epochs=20, verbose=2,sample_weight=weights_train)

      endT_train = time.time()
      T_train=(endT_train-startT_train)/60

      if verbose==True:
        print('It '+str(i)+': training single model took '+format(T_train,'.2f')+' minutes')

      new_bgweight=self.predict_oneModel(data,self.models[i],verbose=verbose)*self.bg_weights[i]
      self.bg_weights.append(new_bgweight)

    endT_trainAll = time.time()
    T_trainAll=(endT_trainAll-startT_trainAll)/60

    if verbose==True:
      print('Training all models  took '+format(T_trainAll,'.2f')+' minutes\n')

      
  def predict_oneModel(self,data,model,verbose=True):
    y_pred=np.ones((1,1))

    startT_test = time.time()

    if self.using_neuralResampler==False:
      y_pred=model.predict_proba(data[:,1:])[:,1]
    else:
      y_pred=model.predict(data[:,1:]).reshape((data.shape[0]))

    endT_test = time.time()
    T_test=(endT_test-startT_test)

    if verbose==True:
      print('Single model prediction took '+format(T_test,'.2f')+' seconds')

    y_pred[y_pred==1]=1-0.0000001
    weights_DR = y_pred/(1-y_pred)
    weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
    weights_DR[weights_DR>1]=1 #some weights blow up
    weights_DR=weights_DR.reshape((data.shape[0],1))
    return weights_DR
  
  def predict(self,data,verbose=True):

    weights_DR_all=np.ones((data.shape[0]))

    startT_testAll = time.time()

    for model in self.models:

      y_pred=np.ones((1,1))

      startT_test = time.time()

      if self.using_neuralResampler==False:
        y_pred=model.predict_proba(data[:,1:])[:,1]
      else:
        y_pred=model.predict(data[:,1:]).reshape((data.shape[0]))

      endT_test = time.time()
      T_test=(endT_test-startT_test)

      if verbose==True:
        print('Single model predciction took '+format(T_test,'.2f')+' s')

      y_pred[y_pred==1]=1-0.0000001
      weights_DR = y_pred/(1-y_pred)
      weights_DR=np.nan_to_num(weights_DR, nan=1, posinf=1, neginf=1)
      weights_DR[weights_DR>1]=1 #some weights blow up
      weights_DR_all=weights_DR_all*weights_DR

    endT_testAll = time.time()
    T_testAll=(endT_testAll-startT_testAll)

    if verbose==True:
      print('all models prediction took '+format(T_testAll,'.2f')+' s\n')

    return weights_DR_all


