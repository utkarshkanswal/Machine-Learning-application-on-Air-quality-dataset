import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os.path as op
from csv import writer
import math
import cmath
import pickle

import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Sequential,load_model
from keras.layers import Input, Embedding
from keras.layers import Dense, Bidirectional
from keras.layers.recurrent import LSTM
import keras.metrics as metrics
import itertools
from tensorflow.python.keras.utils.data_utils import Sequence
from decimal import Decimal
from keras.layers import Conv1D,MaxPooling1D,Flatten,Dense


from keras import backend as K
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




class HybridModel:
    def __init__(self,*args,**kwargs):
        self.r2_score_array=np.empty((8,7),dtype='float32')
    
    def load_lstm(self):
        from keras.models import model_from_json
        json_file = open('LSTM/lstm_tanh.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("LSTM/lstm_tanh.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mse',metrics=['accuracy','mse','mae',rmse])
        return loaded_model
    
    
    def load_gru(self):
        from keras.models import model_from_json
        json_file = open('GRU/gru_tanh_mse.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("GRU/gru_tanh_mse.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mse',metrics=['accuracy','mse','mae',rmse])
        return loaded_model
        
        
    def load_cnn(self):
        from keras.models import model_from_json
        json_file = open('CNN/cnn_relu.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("CNN/cnn_relu.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='mse',metrics=['accuracy','mse','mae',rmse])
        return loaded_model
    
    
    def load_models(self):
        print("Loading XGboost Model.........")
        self.xgboost_model = pickle.load(open("Xgboost/xgboost.sav", 'rb'))
        print("Completed............100%")
        
        print("Loading Randomforest Model.........")
        self.randomforest_model=pickle.load(open("Randomforest/randomforest.sav",'rb'))
        print("Completed............100%")
        
        print("Loading Randomforest Model.........")
        self.knn_model=pickle.load(open("KNN/knn.sav",'rb'))
        print("Completed............100%")
        
        print("Loading Linear Regression Model.........")
        self.linear_model=pickle.load(open("Lasso_or_Linear/linearregression.sav",'rb'))
        print("Completed............100%")
        
        print("Loading Lasso Regression Model.........")
        self.lasso_model=pickle.load(open("Lasso_or_Linear/lasso.sav",'rb'))
        print("Completed............100%")
        
        print("Loading Cnn Model.........")
        self.cnn_model=self.load_cnn()
        print("Completed............100%")
        
        print("Loading LSTM Model.........")
        self.lstm_model=self.load_lstm()
        print("Completed............100%")
        
        print("Loading GRU Model.........")
        self.gru_model=self.load_gru()
        print("Completed............100%")
        
        
    def calculate_r2_score(self,x,y):
        from sklearn.metrics import r2_score
        return r2_score(x,y)
    
    
    def calculate_r2_separately(self,actual,predicted,idx):
        for i in range(0,7):
            self.r2_score_array[idx][i]=self.calculate_r2_score(actual[i],predicted[i])
    
    
    def fit(self,x_test,y_test):
        self.load_models()
        
        self.xgboost_predicted_values=self.xgboost_model.predict(x_test)
        self.randomforest_predicted_values=self.randomforest_model.predict(x_test)
        self.knn_predicted_values=self.knn_model.predict(x_test)
        self.lasso_predicted_values=self.lasso_model.predict(x_test)
        self.linear_predicted_values=self.linear_model.predict(x_test)
        self.lstm_predicted_values=self.lstm_model.predict(x_test)
        self.gru_predicted_values=self.gru_model.predict(x_test)
        self.cnn_predicted_values=self.cnn_model.predict(x_test)
        
        self.calculate_r2_separately(y_test,self.xgboost_predicted_values,0)
        self.calculate_r2_separately(y_test,self.randomforest_predicted_values,1)
        self.calculate_r2_separately(y_test,self.knn_predicted_values,2)
        self.calculate_r2_separately(y_test,self.lasso_predicted_values,3)
        self.calculate_r2_separately(y_test,self.linear_predicted_values,4)
        self.calculate_r2_separately(y_test,self.lstm_predicted_values,5)
        self.calculate_r2_separately(y_test,self.gru_predicted_values,6)
        self.calculate_r2_separately(y_test,self.cnn_predicted_values,7)
        
        return "Done Fitting"
    
    def find_max_r2(self,col):
        
        mx=self.r2_score_array[0,col]
        idx=0
        
        for i in range(0,8):
            if self.r2_score_array[i,col]>mx:
                mx=self.r2_score_array[i,col]
                idx=i
        
        return idx
    
    
    def predict(self,x_test,y_test):
        self.final_predicted_value=np.array(y_test.shape,dtype='float')
        for i in range(1,7):
            idx=self.find_max_r2(i)
            if idx==0:
                self.final_predicted_value[:idx]=self.xgboost_predicted_values[:idx]
            elif idx==1:
                self.final_predicted_value[:idx]=self.randomforest_predicted_values[:idx]
            elif idx==2:
                self.final_predicted_value[:idx]=self.knn_predicted_values[:idx]
            elif idx==3:
                self.final_predicted_value[:idx]=self.lasso_predicted_values[:idx]
            elif idx==4:
                self.final_predicted_value[:idx]=self.linear_predicted_values[:idx]
            elif idx==5:
                self.final_predicted_value[:idx]=self.lstm_predicted_values[:idx]
            elif idx==6:
                self.final_predicted_value[:idx]=self.gru_predicted_values[:idx]
            elif idx==7:
                self.final_predicted_value[:idx]=self.cnn_predicted_values[:idx]
                
        return self.final_predicted_value         