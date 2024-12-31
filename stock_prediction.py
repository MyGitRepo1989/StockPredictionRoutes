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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PredictStocks:
    def __init__(self, stockname):
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data = tsx_data.history(period='70y')
        # Get the last 7 days of closing prices
        self.stockdata = data["Close"][-7:]
        self.data =data["Close"]
        X_shop = data["Close"]
        
    
    
    def predict_stocks(self, model):
        # Fit and transform the data using the scaler
        print("todays data", self.stockdata)
        
        scalerx=StandardScaler()
        scalerx.fit_transform(np.array(self.data[-2000:]).reshape(-1, 1))
        
        scaled_data = scalerx.transform(np.array(self.stockdata).reshape(-1, 1))
        print(scaled_data,"SCALED")
        
        # Predict the next day's closing price
        next_day_prediction = model.predict(scaled_data.reshape(1, -1))
        
        # Inverse transform to get the original scale
        next_day_prediction_original_scale = scalerx.inverse_transform(
            np.array(next_day_prediction).reshape(-1, 1)
        )[0][0]
        
        # Print and return the prediction
        print("NEXT DAY predicted closing price:", next_day_prediction_original_scale)
        
        #seven day prediction
        close_data_7day= scaled_data.copy()
        
        next_seven = []
        for i in range(7):
            # Predict the next day value
            nextday_pred = model.predict(close_data_7day.reshape(1,-1))
            
            # Append the predicted value to the list
            next_seven.append(nextday_pred[0][0])
            
            # Update X_today_scaled for the next iteration
            X_today_scaled_1 = np.append(close_data_7day, nextday_pred[0][0])  # Append the prediction
            X_today_scaled_1 = np.roll(X_today_scaled_1, -1)  # Roll the array to the left by 1
            close_data_7day = X_today_scaled_1[:-1]  # Remove the last element

        # Print the final next_seven predictions
        print("Next seven predictions:", next_seven)
        
         # Inverse transform to get the original scale
        next_seven_original_scale = [scalerx.inverse_transform(i.reshape(-1, 1))[0][0] for i in next_seven]
        print("Next_seven_day_scaled", next_seven_original_scale )
        
        return next_day_prediction_original_scale ,next_seven_original_scale
    
    
    def plot_prediction(self,stockname,nextsevenday ):
        tsx_ticker= stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data= tsx_data.history(period='70y')
      
       
        # Assuming 'alldata' contains the last 10 actual closing prices
        alldata = list(data["Close"][-15:]) + nextsevenday

        # Set the figure size
        plt.figure(figsize=(13, 5))

        # Plot the entire data (historical + predicted)
        plt.plot(range(len(alldata)), alldata, label='Historical + Predicted', marker='o')

        # Plot the predicted values ('next_seven_scaled') starting from the last 7 grid points
        start_index = len(alldata) - len(nextsevenday)
        plt.plot(range(start_index, len(alldata)), nextsevenday, color='red', label='Predicted', marker='o')

        # Annotate values on the plot
        for i, value in enumerate(alldata):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='blue')

        # Add labels, legend, and title
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title(stockname+" - Stock Price Prediction with Historical and Predicted Data")
        plt.legend()
        plt.grid(True)

        plot_path ="static/stock_prediction_plot.jpg"
        plt.savefig(plot_path, format='jpg', dpi=300) 
        return plot_path
        
        
        
  
