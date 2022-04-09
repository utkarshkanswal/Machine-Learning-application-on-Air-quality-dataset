import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os.path as op
from csv import writer
import math
import cmath
import pickle
from sklearn import metrics
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
        self.r2_score_array=np.empty((6,7),dtype='float32')
        self.mse_array=np.empty((6,7),dtype='float32')
        self.rmse_array=np.empty((6,7),dtype='float32')
        self.mae_array=np.empty((6,7),dtype='float32')
        self.index_array=np.empty((6,7),dtype='float32')
        print("Utkarsh")
    
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
            self.r2_score_array[idx][i]=self.calculate_r2_score(actual[:,i],predicted[:,i])

    def calculate_mse_separately(self,actual,predicted,idx):
        for i in range(0,7):
            self.mse_array[idx][i]=metrics.mean_squared_error(actual[:,i],predicted[:,i])      

    def calculate_mae_separately(self,actual,predicted,idx):
        for i in range(0,7):
            self.mae_array[idx][i]=metrics.mean_absolute_error(actual[:,i],predicted[:,i])

    def calculate_rmse_separately(self,actual,predicted,idx):
        for i in range(0,7):
            self.rmse_array[idx][i]=metrics.mean_squared_error(actual[:,i],predicted[:,i])                  
    
    def fit_xgboost(self,x_test):
        return self.xgboost_model.predict(x_test)

    def fit_randomforest(self,x_test):
        return self.randomforest_model.predict(x_test)

    def fit_knn(self,x_test): 
        return self.knn_model.predict(x_test)

    def fit_lasso(self,x_test):
        return self.lasso_model.predict(x_test)
               
    def fit_linear(self,x_test):
        return self.linear_model.predict(x_test)

    def fit_lstm(self,x_test):
        return self.lstm_model.predict(x_test)

    def fit_cnn(self,x_test):
        return self.cnn_model.predict(x_test)

    def fit_gru(self,x_test):
        return  self.gru_model.predict(x_test)

    def fit_machine_learning_model(self,x_test,y_test):
        self.xgboost_predicted_values=self.fit_xgboost(x_test)
        self.randomforest_predicted_values=self.fit_randomforest(x_test)
        self.knn_predicted_values=self.fit_knn(x_test)
        self.linear_predicted_values=self.fit_linear(x_test)
        self.lasso_predicted_values=self.fit_lasso(x_test)
        self.calculate_r2_separately(y_test,self.xgboost_predicted_values,0)
        self.calculate_r2_separately(y_test,self.randomforest_predicted_values,1)
        self.calculate_r2_separately(y_test,self.knn_predicted_values,2)


        self.calculate_mse_separately(y_test,self.xgboost_predicted_values,0)
        self.calculate_mse_separately(y_test,self.randomforest_predicted_values,1)
        self.calculate_mse_separately(y_test,self.knn_predicted_values,2)

        self.calculate_rmse_separately(y_test,self.xgboost_predicted_values,0)
        self.calculate_rmse_separately(y_test,self.randomforest_predicted_values,1)
        self.calculate_rmse_separately(y_test,self.knn_predicted_values,2)

        self.calculate_mae_separately(y_test,self.xgboost_predicted_values,0)
        self.calculate_mae_separately(y_test,self.randomforest_predicted_values,1)
        self.calculate_mae_separately(y_test,self.knn_predicted_values,2)
        
        
    def fit_neural_network_model(self,x_test,y_test):
        self.lstm_predicted_values=self.fit_lstm(x_test)
        self.gru_predicted_values=self.fit_gru(x_test)
        self.cnn_predicted_values=self.fit_cnn(x_test)
        y_test=y_test[:,0]
        self.cnn_predicted_values=self.cnn_predicted_values[:,0]
        self.calculate_r2_separately(y_test,self.lstm_predicted_values,3)
        self.calculate_r2_separately(y_test,self.gru_predicted_values,4)
        self.calculate_r2_separately(y_test,self.cnn_predicted_values,5)

        self.calculate_mse_separately(y_test,self.lstm_predicted_values,3)
        self.calculate_mse_separately(y_test,self.gru_predicted_values,4)
        self.calculate_mse_separately(y_test,self.cnn_predicted_values,5)

        self.calculate_rmse_separately(y_test,self.lstm_predicted_values,3)
        self.calculate_rmse_separately(y_test,self.gru_predicted_values,4)
        self.calculate_rmse_separately(y_test,self.cnn_predicted_values,5)

        self.calculate_mae_separately(y_test,self.lstm_predicted_values,3)
        self.calculate_mae_separately(y_test,self.gru_predicted_values,4)
        self.calculate_mae_separately(y_test,self.cnn_predicted_values,5)
    
    def find_max_r2(self,col):
        
        mx=self.r2_score_array[0,col]
        idx=0
        
        for i in range(0,6):
            if self.r2_score_array[i,col]>mx:
                mx=self.r2_score_array[i,col]
                idx=i
        
        return idx
    
    def find_index_value(self,r2_score,mse,mae,rmse):
        mx=((70*r2_score)-(10*mse+10*mae+10*rmse))/4
        for i in range(1,101):
            for j in range(1,101):
                for k in range(1,101):
                    for l in range(1,101):
                        if i+j+k+l==100:
                            temp_mx=((i*r2_score)-(j*mse+k*mae+l*rmse))/4
                            mx=max(temp_mx,mx)
        return mx
    
    def find_max_index_value(self,col):
        mx=self.find_index_value(self.r2_score_array[0,col],self.mse_array[0,col],self.mae_array[0,col],self.rmse_array[0,col])
        self.index_array[0,col]=mx
        idx=0
        for i in range(0,6):
            temp_index_value=self.find_index_value(self.r2_score_array[i,col],self.mse_array[i,col],self.mae_array[i,col],self.rmse_array[i,col])
            self.index_array[0,col]=temp_index_value
            if temp_index_value>mx:
                idx=i
                mx=temp_index_value
        return idx        

    def predict(self):
        self.final_predicted_value=np.empty((432571, 7),dtype='float')
        for i in range(0,7):
            idx=self.find_max_index_value(i)
            if idx==0:
                self.final_predicted_value[:,i]=self.xgboost_predicted_values[:,i]
            elif idx==1:
                self.final_predicted_value[:,i]=self.randomforest_predicted_values[:,i]
            elif idx==2:
                self.final_predicted_value[:,i]=self.knn_predicted_values[:,i]
            elif idx==3:
                self.final_predicted_value[:,i]=self.lstm_predicted_values[:,i]
            elif idx==4:
                self.final_predicted_value[:,i]=self.gru_predicted_values[:,i]
            elif idx==5:
                self.final_predicted_value[:,i]=self.cnn_predicted_values[:,i]
                
        return self.final_predicted_value         