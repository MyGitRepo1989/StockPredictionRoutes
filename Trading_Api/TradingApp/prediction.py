import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')
import random
import mplcyberpunk
import datetime
import dotenv
import os

dotenv.load_dotenv()
PLOT_PATH = os.getenv("STOCKS_PLOT_PATH")
ADJUSTED_PLOT_PATH = os.getenv("STOCKS_ADJUSTED_PLOT")



class PredictStocks:
    def __init__(self, stockname):
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        
        print("Got yfinance data ",tsx_data)
        data = tsx_data.history(period='2y')
        # Get the last 7 days of closing prices
        #TRY  **data["Close"][-8:-1]
        #self.stockdata = data["Close"][-7:]
 
        self.stockdata = data["Close"][-8:-1]
        self.data =data["Close"]
        X_shop = data["Close"]
        
        
    
    
    def predict_stocks(self, model):
        # Fit and transform the data using the scaler
        print("todays data", self.stockdata)
        
        scalerx=StandardScaler()
        scalerx.fit_transform(np.array(self.data[-300:]).reshape(-1, 1))
        
        scaled_data = scalerx.transform(np.array(self.stockdata).reshape(-1, 1))
        print(scaled_data,"Scaled data")
        
        # Predict the next day's closing price
        next_day_prediction = model.predict(scaled_data.reshape(1, -1))
        
        # Inverse transform to get the original scale
        next_day_prediction_original_scale = scalerx.inverse_transform(
            np.array(next_day_prediction).reshape(-1, 1)
        )[0][0]
        
        # Print and return the prediction
        print("NEXT DAY PREDICTION ", next_day_prediction_original_scale)
        
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
        print("NEXT SEVEN DAY PREDICTIONS", next_seven)
        
         # Inverse transform to get the original scale
        next_seven_original_scale = [scalerx.inverse_transform(i.reshape(-1, 1))[0][0] for i in next_seven]
        print("NEXT SEVEL DAY NOT SCALED", next_seven_original_scale )
        
        last15_predictions = []
        # Iterate through the last 15 days
        for i in range(15):
            # Define the window for the 7 previous days
            start_idx = -(16 - i + 7)
            end_idx = -(16 - i)
            input_data = np.array(self.data[start_idx:end_idx]).reshape(-1, 1)
            
            
            # Transform the input data using the scaler
            scaled_input = scalerx.transform(input_data).reshape(1, 7, 1)
            print(scaled_input.shape,"SCALED INPUT SHAPE")
            # Make the prediction for the day
            pred =  model.predict(scaled_input.reshape(1, -1))
            print(pred ,"PREDICTION NOT SCALED")
            # Apply inverse transform to get the actual prediction
            actual_pred = scalerx.inverse_transform(pred)[0][0]
            last15_predictions.append(actual_pred)
            
        print(last15_predictions ,"LAST 15 DAYS")
        
        return next_day_prediction_original_scale ,next_seven_original_scale, last15_predictions
   
    
    def news_adjusted(self,stockname,last15_predictions,nextsevenday,news_influence):
        print(stockname,last15_predictions)
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data = tsx_data.history(period='1y')
        print(data)
        
        historical_data = list(data["Close"][-15:])
        print(historical_data,"HISTORIC")
        actual_values= historical_data
        predictions = last15_predictions
        errors = [actual_values[i] - predictions[i] for i in range(len(predictions))]
        print("errors", errors)
        mean_diff = np.mean(errors)

        adjusted_predictions = [
            pred + (err * random.uniform(1.2, 1.5) if abs(err) > abs(mean_diff) else err * random.uniform(0.5, 1.0))
            for pred, err in zip(predictions, errors)]
        
      
        decay_factors = [1.1, .8, .7, 0.5, 0.3, 0.2, 0.1]  # Strongest influence on first 3 days, fading off

        nextsevenday_adjusted = [
            pred * (1 + (news_influence / 100) * decay) for pred, decay in zip(nextsevenday, decay_factors)
            ]

        print("7 day adjusted values",nextsevenday_adjusted , "/n historic adjusted",adjusted_predictions )
        
        return adjusted_predictions, nextsevenday_adjusted 
    
    
    '''
    def plot_prediction(self, stockname, nextsevenday, last15_predictions):
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data = tsx_data.history(period='2y')
        
        #last15_predictions =last15_predictions[1:]
        # Extract historical data (last 15 days) and append next 7 days predictions
        historical_data = list(data["Close"][-15:])
        all_data = historical_data + nextsevenday  # Historical + new predictions
             
        # Set the figure size
        plt.figure(figsize=(13, 5))

        # Plot historical data
        plt.plot(range(len(historical_data)), historical_data, label='Historical', marker='o', color='blue')

        # Plot the last 15 predictions
        plt.plot(range(len(historical_data)), last15_predictions, label='Last 15 Predictions', marker='x', color='green')

        # Plot the next 7 predictions
        start_index = len(historical_data)
        plt.plot(range(start_index, len(all_data)), nextsevenday, label='Next 7 Predictions', marker='o', color='red')

        # Annotate values on the plot
        for i, value in enumerate(historical_data):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='blue')
        for i, value in enumerate(last15_predictions):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='top', color='green')
        for i, value in enumerate(nextsevenday):
            plt.text(start_index + i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='red')

        # Add labels, legend, and title
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title(stockname + " - Stock Price Prediction with Historical and Predicted Data")
        plt.legend()
        plt.grid(True)
        
        print("Plot created.")

        # Save the plot
        plot_path = "static/stock_prediction_plot.jpg"
        plt.savefig(plot_path, format='jpg', dpi=300)         
        return plot_path
    '''
    
    def plot_prediction(self, stockname, nextsevenday, last15_predictions):

        ##190748 vtrader bg
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data = tsx_data.history(period='2y')
        
        #last15_predictions =last15_predictions[1:]
        # Extract historical data (last 15 days) and append next 7 days predictions
        historical_data = list(data["Close"][-15:])
        all_data = historical_data + nextsevenday  # Historical + new predictions
        
        plt.clf()     
        # Set the figure size
        plt.figure(figsize=(13, 5))
        plt.subplots_adjust(top=0.85)
      
        
        plt.style.use("cyberpunk")
        
        plt.rcParams['axes.facecolor'] = '#190748'
        # Plot historical data
        plt.plot(range(len(historical_data)), historical_data, label='Historical', marker='o', color='#00f0ff') #blue

        # Plot the last 15 predictions
        plt.plot(range(len(historical_data)), last15_predictions, label='Last 15 Predictions', marker='x', color='#5fff82') #green

        # Plot the next 7 predictions
        start_index = len(historical_data)
        plt.plot(range(start_index, len(all_data)), nextsevenday, label='Next 7 Predictions', marker='o', color='#a382ff') #purple
        
        
        # Add labels, legend, and title
        plt.xlabel("Time", fontsize=15, color='white', fontweight=900)
        plt.ylabel("Stock Price",fontsize=15, color='white', fontweight=900)
        plt.title("\n" + stockname + " - Stock Price Prediction with Historical and Predicted Data\n", fontsize=19, color='white', fontweight=900)
        
        plt.legend(loc='upper left')
        plt.grid(True)
        mplcyberpunk.add_glow_effects()
        
        
         # Annotate values on the plot
        for i, value in enumerate(historical_data):
            plt.text(i, value, f"{value:.2f}", fontsize=10, ha='center', va='bottom', color='#affaff',fontweight=400) #blue
        for i, value in enumerate(last15_predictions):
            plt.text(i, value, f"{value:.2f}", fontsize=10, ha='center', va='top', color='#1fff50',fontweight=400) #green
        for i, value in enumerate(nextsevenday):
            plt.text(start_index + i, value, f"{value:.2f}", fontsize=10, ha='center', va='bottom', color='#d7caf9',fontweight=400) #purple
        
        
        #VERTICLE LINE
        plt.axvline(x=14, color='white', linestyle='-', linewidth=.3, alpha=1)
        
        # Get today's date in the format 'DD-MM-YYYY'
        today_plot_date = datetime.datetime.now().strftime('%d-%m-%Y')
        # Update the plt.text line
        plt.text(14, plt.ylim()[0], "Today: " +today_plot_date, color='white', ha='center', va='bottom', fontsize=10, fontweight=900) #date

        print("Plot created.")

        # Save the plot
        plot_path = PLOT_PATH
        plt.savefig(plot_path, format='jpg', dpi=300,facecolor='#190748')         
        return plot_path
    
    

    def prediction_eval(self, nextday, nextsevenday, stockname):
        print(nextday, nextsevenday, stockname)
        try: 
            ticker = yf.Ticker(stockname)
            df_eval = ticker.history(period="2y")
            print(df_eval)
            
            #DIFF $ Calculation
            stock_mean_difference_price =df_eval["Close"][-30:].diff().mean()
            stock_differnce_closing_price = df_eval["Close"][-1:][0] + (stock_mean_difference_price)
            print(stock_differnce_closing_price)
            
            #VOLITILITY PRICE Calculation
            stock_volility = np.std(df_eval["Close"][-30:].pct_change())*100
            stock_volility_closing_price = df_eval["Close"][-1:][0] * stock_volility/100 
            print(stock_volility_closing_price)
            
            
            #difference in prediction
            predicted_difference = nextday - df_eval["Close"][-1:][0]

            # Predicted stock prices for 7 days
            y = np.array(nextsevenday)
            x = np.arange(1, len(y) + 1).reshape(-1, 1)  # Days (1 to 7)

            # Linear Regression Trendline
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]  # Slope of the trendline

            # Determine trend direction and strength
            direction = "Upward 📈" if slope > 0 else "Downward 📉" if slope < 0 else "Sideways ➡️"
            print(f"Trend: {direction}, Strength: {slope:.3f} per day")
            
            print(direction,slope )
        except Exception as e:
            print (e)
        return stock_mean_difference_price, stock_volility_closing_price, direction,slope ,predicted_difference



    def plot_prediction_adjustment(self, stockname, nextsevenday, last15_predictions, adjusted_predictions,nextsevenday_adjusted ):
        
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        data = tsx_data.history(period='2y')
        
        #last15_predictions =last15_predictions[1:]
        # Extract historical data (last 15 days) and append next 7 days predictions
        historical_data = list(data["Close"][-15:])
        all_data = historical_data + nextsevenday  # Historical + new predictions
        
        # Set the figure size
        plt.figure(figsize=(13, 5))

        # Plot historical data
        plt.plot(range(len(historical_data)), historical_data, label='Historical', marker='o', color='blue')

        # Plot the last 15 predictions
        plt.plot(range(len(historical_data)), last15_predictions, label='Last 15 Predictions', marker='x', color='green')
        
        # Plot the adjustments
        plt.plot(range(len(historical_data)), adjusted_predictions, label='Last 15 News Score', marker='x', color='#6e00ff', linewidth=2, alpha=0.6)

        # Plot the next 7 predictions
        start_index = len(historical_data)
        plt.plot(range(start_index, len(all_data)), nextsevenday, label='Next 7 Predictions', marker='o', color='red')
        
        # Plot the next 7 adjusted
        start_index = len(historical_data)
        plt.plot(range(start_index, len(all_data)), nextsevenday_adjusted , label='Next 7 News Score', marker='o', color='#6e00ff', linewidth=2, alpha=0.6)
        

        # Annotate values on the plot
        for i, value in enumerate(historical_data):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='blue')
        for i, value in enumerate(last15_predictions):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='top', color='green')
        for i, value in enumerate(adjusted_predictions):
            plt.text(i, value, f"{value:.2f}", fontsize=8, ha='center', va='top', color='#6e00ff')
            
        for i, value in enumerate(nextsevenday):
            plt.text(start_index + i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='red')
        for i, value in enumerate(nextsevenday_adjusted):
            plt.text(start_index + i, value, f"{value:.2f}", fontsize=8, ha='center', va='bottom', color='#6e00ff')
            

        
        # Add labels, legend, and title
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title(stockname + " - Stock Price Prediction with Historical and Predicted Data")
        plt.legend()
        plt.grid(True)
        print("PLOT CREATED")

        # Save the plot
        adjusted_plot_path = ADJUSTED_PLOT_PATH
        plt.savefig(adjusted_plot_path, format='jpg', dpi=300)         
        return adjusted_plot_path
   

