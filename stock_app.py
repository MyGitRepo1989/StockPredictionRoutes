from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import stock_model_training 
import stock_prediction
from tensorflow.keras.models import load_model
from stock_model_training import *
from stock_prediction import *
from flask import Flask, request, jsonify, render_template_string
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    index_html = """
    <!doctype html>
    <html lang="en">
        <head>
            <title>Stock Prediction</title>
        </head>
        <body>
            <h1>Welcome to the Stock Prediction App!</h1>
            <form action="/stock_prediction" method="get">
                <label for="stocksymbol">Enter Stock Symbol:</label>
                <input type="text" id="stocksymbol" name="stocksymbol" required>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return render_template_string(index_html)



@app.route('/stock_prediction')
def stock_prediction():
    print("Request args:", request.args)  # Debug statement to print query params
    stocksymbol = request.args.get('stocksymbol', None)
    if not stocksymbol:
        return "Error: Stock symbol is required!", 400
    
    print("Received stock symbol:", stocksymbol)
   
    
    try:
        #comment start - training if the model already exists
        
        # Use `TrainStockModel` to generate data
        trainstockmodel= TrainStockModel()
        x_train, x_test, y_train, y_test, scalerx = trainstockmodel.makedata(stocksymbol)
        print("Train/Test Split and Scaler Data:")
        print("x_train:", x_train, "x_test:", x_test, "y_train:", y_train, "y_test:", y_test, "scalerx:", scalerx)
        
        # train model we find 500 epochs is a tested best parameter 
        stock_model = trainstockmodel.train_model(x_train, x_test, y_train, y_test,epochs=500)
        model_path= "/Volumes/iMac2024/2024_nmodes/clustering_2024/news_flask_app/stock_models/" + stocksymbol+".h5"
        print(model_path)
        stock_model.save(model_path)
        
        #comment end - here 

        #load model
        model_path= "/Volumes/iMac2024/2024_nmodes/clustering_2024/news_flask_app/stock_models/" + stocksymbol+".h5"
        stock_model_loaded = load_model(model_path)
        
        
        #Initialize PredictStocks with the stockname
        stockpredictor = PredictStocks(stocksymbol)

        # Use the PredictStocks method to predict the next day
        nextday, nextsevenday = stockpredictor.predict_stocks(stock_model_loaded)
        
        #plot the graph and save it
        plot_path = stockpredictor.plot_prediction(stocksymbol,nextsevenday)
        print("Next day's predicted price:", nextday)
        print(plot_path)

   
        # Return the prediction as a JSON response and include the path to the image
        filename = os.path.basename(plot_path)
        return render_template_string("""
        <html lang="en">
        <head>
            <title>Stock Prediction</title>
        </head>
        <body>
            <h1>Stock Prediction for {{ stocksymbol }}</h1>
            <p>Next day's predicted price: {{ nextday }}</p>
            <p>7-day prediction: {{ sevenday }}</p>
            <p>{{filename}}</p>
            <img src="{{ url_for('static', filename=filename) }}" style="width: 80%; display: block; margin: 0 auto;">
        </body>
    </html>
        """, stock_symbol=stocksymbol, nextday=nextday, sevenday=nextsevenday, filename=filename)
        
        
    except Exception as e:
        # Handle any errors during data processing
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)



