
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
import openai
from openai import OpenAI

dotenv.load_dotenv()

STOCK_MODELS_DIR = os.getenv("STOCK_MODELS_DIR")
STOCK_PREDICTION_IMG = os.getenv("STOCK_PREDICTION_IMG")
STOCKS_PLOT_PATH = os.getenv("STOCKS_PLOT_PATH")
openai.api_key = os.getenv("OPEN_AI_KEY")
text_file_path = os.environ.get("TEXT_FILE_PATH")



class PredictStocks:
    def __init__(self, stockname):
        
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        
        print("GOT DATA",tsx_data)
        data = tsx_data.history(period='2y')
 
        #self.stockdata = data["Close"][-8:-1]
        self.stockdata = data["Close"][-8:-1]
        self.data =data["Close"]/1000
        X_shop = data["Close"]
        
        
    
    
    def predict_stocks(self, model):
        # Fit and transform the data using the scaler
        print("todays data", self.stockdata)
        print("SHAPE TODAYS DATA", self.data.shape)
        
        scalerx=StandardScaler()
        scalerx.fit_transform(np.array(self.data[-300:]).reshape(-1, 1))
        
        scaled_data = scalerx.transform(np.array(self.stockdata).reshape(-1, 1))
        print(scaled_data,"SCALED DATA")
        
        data_last_7days= scaled_data[-8:-1]
        print(data_last_7days,"LAST 7 Dayss")
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
        
        '''
        # DO THIS FOR APPLE STOCKS NOT COINS
        
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
        
        '''
        
        # USE THIS FOR COINS 
        next_seven = []
        for i in range(7):
            # Predict the next day value (still in scaled form)
            nextday_pred = model.predict(close_data_7day.reshape(1, -1))
            
            # Append the scaled prediction (as array) to next_seven
            next_seven.append(nextday_pred)

            # Update the input sequence with the scaled predicted value
            close_data_7day = np.append(close_data_7day, nextday_pred[0][0])  # Append prediction
            close_data_7day = np.roll(close_data_7day, -1)                    # Shift left
            close_data_7day = close_data_7day[:-1]                        # Trim to original length

        next_seven = []
        for i in range(7):
            # Get the last 7 values from self.data including any previous predictions
            last_7_raw = np.array(self.data[-7:]).reshape(-1, 1)

            # Scale them
            scaled_input = scalerx.transform(last_7_raw).reshape(1, -1)

            # Predict
            pred_scaled = model.predict(scaled_input)
            
            # Inverse transform to get actual value
            pred_actual = scalerx.inverse_transform(pred_scaled)[0][0]

            # Append to results and to the raw data stream
            next_seven.append(pred_actual)
            self.data = np.append(self.data, pred_actual)
        
        
        print(next_seven)
        # Print the raw scaled predictions (optional)
        #print("Next seven predictions (scaled):", [pred[0][0] for pred in next_seven])

        # Inverse transform all 7 predictions at once
        #next_seven_original_scale = scalerx.inverse_transform(np.array(next_seven).reshape(-1, 1)).flatten().tolist()
        next_seven_original_scale = next_seven
        print("Next_seven_day_scaled:", next_seven_original_scale)

        
        
        last15_predictions = []
        # Iterate through the last 15 days
        for i in range(15):
            # Define the window for the 7 previous days
            start_idx = -(16 - i + 7)
            end_idx = -(16 - i)
            input_data = np.array(self.data[start_idx:end_idx]).reshape(-1, 1)
            
            
            # Transform the input data using the scaler
            scaled_input = scalerx.transform(input_data).reshape(1, 7, 1)
            print(scaled_input.shape," SHAPE OF STOCK")
            # Make the prediction for the day
            pred =  model.predict(scaled_input.reshape(1, -1))
            print(pred ,"PREDICTIONS")
            # Apply inverse transform to get the actual prediction
            actual_pred = scalerx.inverse_transform(pred)[0][0]
            last15_predictions.append(actual_pred)
            
        print(last15_predictions ,"LAST 15 DAYS")
        
        return next_day_prediction_original_scale ,next_seven_original_scale, last15_predictions
   
    
    
    
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
        
        '''
        #THIS IS THE OLDER GRAPH STYLE
         # Annotate values on the plot
        for i, value in enumerate(historical_data):
            plt.text(i, value + 0.05, f"{value:.2f}", fontsize=13, ha='center', va='bottom', color='#affaff',fontweight=900) #blue
        for i, value in enumerate(last15_predictions):
            plt.text(i, value + 0.05, f"{value:.2f}", fontsize=13, ha='center', va='top', color='#1fff50',fontweight=900) #green
        for i, value in enumerate(nextsevenday):
            plt.text(start_index + i, value+ 0.05, f"{value:.2f}", fontsize=13, ha='center', va='bottom', color='#d7caf9',fontweight=900) #purple
        
        '''
        
        # Annotate values on the plot
        all_data = np.concatenate([historical_data, last15_predictions, nextsevenday])
        shift = 0.02 * (np.max(all_data) - np.min(all_data))
        
        for i, value in enumerate(historical_data):
            plt.text(i, value + shift , f'{int(value)}', fontsize=9, ha='center', va='bottom', color='#00baff', fontweight=400, 
                    bbox=dict(facecolor='#190748', alpha=0.8, edgecolor='none',pad=1)) #blue
        for i, value in enumerate(last15_predictions):
            plt.text(i, value + shift , f'{int(value)}', fontsize=9, ha='center', va='bottom', color='#00ff53', fontweight=400, 
                    bbox=dict(facecolor='#190748', alpha=0.8, edgecolor='none',pad=1)) #green
        for i, value in enumerate(nextsevenday):
            plt.text(start_index + i, value + shift , f'{int(value)}', fontsize=9, ha='center', va='bottom', color='#b596ff', fontweight=400, 
                    bbox=dict(facecolor='#190748', alpha=0.8, edgecolor='none',pad=1)) #purple
        
        
        #VERTICLE LINE
        plt.axvline(x=14, color='white', linestyle='-', linewidth=.3, alpha=1)
        
        # Get today's date in the format 'DD-MM-YYYY'
        today_plot_date = datetime.datetime.now().strftime('%d-%m-%Y')
        # Update the plt.text line
        plt.text(14, plt.ylim()[0], "Today: " +today_plot_date, color='white', ha='center', va='bottom', fontsize=10, fontweight=900) #date

        print("Plot created.")

        # Save the plot YOU CAN CHANGE THIS TO WHERE EVER YOU WANT TO WRITE THE GRAPH
        plot_path = STOCKS_PLOT_PATH 
        plt.savefig(plot_path, format='jpg', dpi=300,facecolor='#190748')         
        return plot_path
    
    
    def prediction_summary(self, stockname, nextsevenday, last15_predictions,nextday):
        tsx_ticker = stockname
        tsx_data = yf.Ticker(tsx_ticker)
        print("GOT DATA",tsx_data)
        data = tsx_data.history(period='2y')
        data_volitility= data["Close"][-60:]
        print(data_volitility, data_volitility.shape)
        
        max_last60_days = data_volitility.max()
        min_last60_days = data_volitility.min()
        print("MAX MIN", max_last60_days, min_last60_days)
        
        #calculate last 60 day volitility
        log_returns_60 = np.diff(np.log(data_volitility))
        vol_60 = np.std(log_returns_60)
        
        print("Volitility 60 days" , vol_60)
        
        #calculate last next 7 day volitility
        log_returns_7 = np.diff(np.log(nextsevenday))
        vol_7 = np.std(log_returns_7)
        
        print("Volitility 7 days", vol_7)
        
        #calculate change in volitility 
        change_percent = ((vol_7 - vol_60) / vol_60) * 100
        print ( "Volitility " , round(change_percent,2))
        
        # Slope 
        x = np.arange(7).reshape(-1, 1)
        y = np.array(nextsevenday)

        model_slope = LinearRegression().fit(x, y)
        slope = model_slope.coef_[0]
        print("Slope", round(slope,2))
        

        client = OpenAI()

        # Prompt construction
        prompt = (
            f'Write a consise analysis between 2- 3 lines to explain stock prediction statistically \
                mention the highest max price and lowest price, mention volitility and the slope '
                
            f"The {stockname} stock has a 60-day volatility of {round(vol_60, 2)} "
            f"with a predicted 7-day volatility of {round(vol_7, 2)}, marking a volatility change of {round(change_percent, 2)}%. "
            f"The maximum price in the last 60 days was {round(max_last60_days, 2)}, and the minimum was {round(min_last60_days, 2)}. "
            f"The 7-day slope is {round(slope, 2)}, and the next day predicted price is {round(nextday, 2)}."
        )



        # Chat model call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )


        #print(response.choices[0].message.content.strip())
        response = response.choices[0].message.content.strip()
        
        try:
            with open(text_file_path , "w") as f:
                f.write(str(response))
 
        except Exception as e:
            print("File Save Error:",e)
        
        return response , text_file_path

        
        
        
# below is if you have to test this file individually

'''
if __name__ =="__main__":
    stock_model_loaded = load_model("stock_models/ETH-USD.h5")
    stockname = "ETH-USD"
    Pred_Class =  PredictStocks(stockname)
    nextday, nextsevenday,last15_predictions = Pred_Class.predict_stocks( stock_model_loaded)
    response = Pred_Class.prediction_summary(stockname, nextsevenday, last15_predictions,nextday)
    print(response)

'''
