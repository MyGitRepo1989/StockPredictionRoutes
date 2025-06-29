
import sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from stock_predictor_1 import *
from stock_predictor_1 import PredictStocks
import os
import dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dotenv.load_dotenv()

STOCK_MODELS_DIR = os.getenv("STOCK_MODELS_DIR")
STOCK_PREDICTION_IMG = os.getenv("STOCKS_PLOT_PATH")


def get_predictions(stocksymbol):

    model_path = f"{STOCK_MODELS_DIR}/{stocksymbol}.h5"
    
    stock_model_loaded = load_model(model_path)
    
    # Initialize PredictStocks with the stock name
    stockpredictor = PredictStocks(stocksymbol)
    
    # Use PredictStocks method to predict the next day
    nextday, nextsevenday,last15_predictions = stockpredictor.predict_stocks(stock_model_loaded)
    
    plot_path = stockpredictor.plot_prediction(stocksymbol, nextsevenday,last15_predictions)
    summary , text_file_path = stockpredictor.prediction_summary(stocksymbol, nextsevenday, last15_predictions,nextday)
    return nextday, nextsevenday,last15_predictions, plot_path , summary , text_file_path



if __name__ =="__main__":
    
    stocksymbol = "BTC-USD"
    #stockpredictor = PredictStocks(stocksymbol)
    nextday, nextsevenday,last15_predictions, plot_path, summary , text_file_path = get_predictions(stocksymbol)
    
    print("Next day's predicted price:", nextday)
    print("Next 7 days' predicted prices:", nextsevenday)
    print("Last 15 days' predicted prices:", last15_predictions)
    print("this is the summary text string ", summary )
    print("the summary text file is saved in this location",  text_file_path)
    
    print(plot_path)
    
