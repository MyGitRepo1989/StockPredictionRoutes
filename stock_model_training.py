from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import stock_model_training 
import stock_prediction


class TrainStockModel:
    def __init__(self) -> None:
        pass
      
    def makedata(self,stocksymbol):
        tsx_ticker= stocksymbol
        tsx_data = yf.Ticker(tsx_ticker)
        data= tsx_data.history(period='70y')
        print(data.head(2))
        data.reset_index(inplace=True, drop=False)

        X_shop = data["Close"]
        #shift y
        y_shop= data["Close"].shift(-8)
        y_shop = y_shop.interpolate(method='linear') #fill up nans

        X_shop2= X_shop[-2000:]
        y_shop2= y_shop[-2000:]

        scalery=StandardScaler()
        scalerx=StandardScaler()

        X_scaled=scalerx.fit_transform(np.array(X_shop2).reshape(-1, 1))
        y_scaled=scalery.fit_transform(np.array(y_shop2).reshape(-1, 1))

        stock_scaled=X_scaled
        k=[]
        stock_series_s=[]
        c=0
        for s in range(len(stock_scaled)-6):    
            for i in range(7):    
                k.append(stock_scaled[c][0]) 
                c=c+1
            stock_series_s.append(k) 
            c=c-6
            k=[]
        print(stock_series_s)

        print(len(stock_series_s),y_scaled.shape)
        y_scaled_1= y_scaled[:len(stock_series_s)] #match len of series to y

        #establish train test size
        data_to_use= round((y_scaled_1.shape[0]*5)/100) #5 % data
        print( "training data", data_to_use )

        y_test= y_scaled_1[:data_to_use]
        y_train =y_scaled_1[data_to_use:]
        print("Y train test", y_train.shape,y_test.shape)

        x_test= stock_series_s[:data_to_use]
        x_train =stock_series_s[data_to_use:]
        
        print("X train test",len(x_train),len(x_test))
        x_train=np.array(x_train)
        x_test=np.array(x_test)
        
        return x_train, x_test, y_train, y_test, scalerx
    
    
    def train_model(self,x_train, x_test, y_train, y_test, epochs):
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate,clipnorm=1.0)
        model_basic=Sequential()
        model_basic.add(LSTM((100), batch_input_shape=(None,7,1), return_sequences=True,dropout=0))
        model_basic.add(LSTM((50), batch_input_shape=(None,7,1), return_sequences=True,dropout=0.0))
        model_basic.add(LSTM((20), batch_input_shape=(None,7,1), return_sequences=True,dropout=0.0))
        model_basic.add(LSTM((10), batch_input_shape=(None,7,1), return_sequences=True,dropout=0.0))
        model_basic.add(LSTM((5), batch_input_shape=(None,7,1), return_sequences=False,dropout=0.0))

        model_basic.add(Dense((1)))

        model_basic.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mse'])
        history = model_basic.fit(x_train, y_train, epochs=epochs,
        validation_data=(x_test, y_test))
        
        return model_basic
        