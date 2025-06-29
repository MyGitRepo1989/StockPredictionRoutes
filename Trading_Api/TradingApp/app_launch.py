from flask import Flask, request, render_template_string, request
from flask import Flask, render_template, redirect, url_for
from flask import url_for
import sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import stock_prediction
from keras.models import load_model
from stock_prediction import PredictStocks

import stock_app_display_ind
from stock_app_display_ind import calculate_indicators
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate

import dotenv
import os
dotenv.load_dotenv()

OPENAIKEY= os.getenv("STOCKS_OPENAIKEY")
MODELPATH = os.getenv("STOCKS_MODELPATH")
PLOT_PATH = os.getenv("STOCKS_PLOT_PATH")


os.environ["OPENAI_API_KEY"] = OPENAIKEY
llm = ChatOpenAI(temperature=0.5)

app = Flask(__name__)

template = '''

<html>
<style>
      body {

        justify-content: center;
        align-items: center;
        min-height: 100vh; /* Full viewport height */
        margin_top :2%;
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
      }
      form {
        margin-left: 10%;
        background: #ffffff;
        padding: 20px;
        margin-right: 10%;
      }

      h1 {
        margin-left: 10%;
        margin-bottom: -10px;
      }

      label {
        display: block;
        margin-bottom: 0px;
        font-weight: bold;
        color: #555;
      }

      button {
        margin-top: -10px;
        padding: 10px 20px;
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
      }

      button:hover {
        background-color: #0056b3;
      }
 
      .stockresults {
        margin-left: 10%;
        margin-right: 10%;   
        display: flex;
      justify-content: space-between; /* Adjust spacing */
      gap: 20px;

      }
      
      .stockgraph {
        margin-left: 10%;
        margin-right: 10%;   
      }
      
      .colleft {
        width:70%;
    flex-direction: column;
      }
      
      .colright {
        width:30%;
         display: flex;
    flex-direction: column;
      }
  
    .graphplot {
        margin-left: -0% !important;
        height: 50%;
      }
      
    .smallinfo{
      font-size:13px;
      font-weight:300;
      margin-top: -15px !important;
      padding-top: -3px !important;
      width:90%;
    }
 
    .link-style {
    color: blue; /* Sets text color */
    font-size:13px;
      }

    .link-style:visited {
        color: blue; /* Ensures visited links stay blue */
    }

    .link-style:hover {
        color: darkblue; /* Changes color on hover */
    }
    
    .link-style img {
    width: 16px; /* Standard favicon size */
    height: 16px;
    }   

  .titlecol{
    margin-top:10px!important;
    margin-bottom:0px!important;
  }
  </style>

  </head>

  <body>
  
    <h1>Predictive Stock Model</h1>
    <br>
    <form method="post">
      <label for="stocksymbol">Select Stock Symbol:</label><br>
      <select id="stocksymbol" name="stocksymbol">
        <option value="AAPL">Apple (AAPL)</option>
        <option value="MSFT">Microsoft (MSFT)</option>
        <option value="TSLA">Tesla (TSLA)</option>
        <option value="BCE.TO">Bce Toronto (BCE.TO)</option>
      </select><br><br>
      <button type="submit">Submit</button>
    </form>
      
    {% if result %}
    <div class ="stockresults">
    
        
        
        <div class="colleft">
        <h2 class="titlecol">Result for selected stock: {{ stocksymbol }}</h2>
        <p> <a class= "link-style" href="{{ url_for('route2', stocksymbol=stocksymbol) }}">AI Indicators (Takes few secounds)</a> </p>
        
        <p> 7-Day Trend : {{direction}} Slope (Strength) of Trend :{{str_slope}} </p>
        <p class= "smallinfo">7-day prediction: {{ sevenday }}</p>
        <p class= "smallinfo">Next day's predicted price: {{ nextday }}</p>
        <p> Stock Statistical Analysis: </p>
        <p class= "smallinfo"> Price Change: {{ str_predicted_difference }} Historic Average Daily Price Change: {{ str_stock_mean_difference_price }} Price Volatility (Statistical Measure of Risk): {{ str_stock_volility_closing_price}} </p>
        <p class= "smallinfo" > {{data_analysis}} </p>
        <!--<p class= "smallinfo">Historic Average Daily Price Change: {{ str_stock_mean_difference_price }}</p> -->
        <!--<p class= "smallinfo">Price Volatility (Statistical Measure of Risk): {{ str_stock_volility_closing_price}}</p>-->
        

        </div>
        
        <div class="colright">
        <h2 class="titlecol">Top Headlines </h2>
        <p> News Influence: {{ News_Trend }}</p>
             
          <a href="https://www.cnn.com/2025/02/07/economy/us-jobs-report-january-final/index.html" 
        target="_blank" class="link-style">
        <img src="https://www.cnn.com/favicon.ico" alt="CNN"> Read Article: US Jobs Report
          </a>
          
          <a href="https://www.cnn.com/2025/01/29/economy/fed-rate-decision-january/index.html" 
        target="_blank" class="link-style">
        <img src="https://www.cnn.com/favicon.ico" alt="CNN"> Read Article: Fed Rate Decision
          </a>
          
          
         <a href="https://apnews.com/article/trump-tariffs-mexico-canada-71761a2894e13a050717afda4fd8131a" 
        target="_blank" class="link-style">
        <img src="https://www.apnews.com/favicon.ico" alt="CNN"> Read Article: Trump Tarrifs Plans
          </a>
          
          <a href=https://edition.cnn.com/2025/02/28/investing/us-stocks-nasdaq/index.html" 
        target="_blank" class="link-style">
        <img src="https://www.cnn.com/favicon.ico" alt="CNN"> Read Article: Markets Close in Red
          </a>
          
          <a href="https://edition.cnn.com/2025/02/27/economy/bird-flu-egg-prices-higher/index.html" 
        target="_blank" class="link-style">
        <img src="https://www.cnn.com/favicon.ico" alt="CNN"> Read Article: Trump's Bird Flu Plan
          </a>
          
          
         <a href="https://www.cnbc.com/2025/03/03/auto-giants-scramble-to-suffer-the-least-amid-trump-tariff-threats.html" 
        target="_blank" class="link-style">
        <img src="https://www.cnbc.com/favicon.ico" alt="CNN"> Read Article: Auto Giants Suffer
          </a>
        
        <p> 312+ more articles </p>
        
        
        
        </div>
     
        <!-- <p> filename :{{filename}}</p> -->
        
 
   
    </div>
    
    <div class ="stockgraph">
    
       <a href="#" class= "link-style" onclick="showImage('.jpg')">Stock Prediction</a>  | 
        <a href="#" class= "link-style" onclick="showImage('.jpg')">News Influence Score</a>  | 
        <a href="#" class= "link-style" onclick="showImage('.jpg')">Social Influence Score</a>  | 
        <a href="#" class= "link-style" onclick="showImage('.jpg')">Financials Influence Score</a>  
        
        <img class="graphplot" src="./static/.jpg" style="width: 90%;">
    {% endif %}

    </div>
    
    <script>
        function showImage(filename) {
            document.querySelector(".graphplot").src = "./static/" + filename + "?t=" + new Date().getTime();
        }
    </script>
 
  </body>
</html>
'''

def llm_analysis(data):
  # Define the advisory-style summarization prompt
  prompt_template = PromptTemplate(
  input_variables=["data"],
  template="""
              You are a statistician for financial stocks. Compare the following:


              1. Predicted Price Change vs. Historic Average Daily Price Change.
              2. Analyze Price Volatility and its implications.
              3. Examine the 7-Day Trend and Slope.
              4. Summarize your findings in 3-4 sentences.


              Example response:
              Analysis:

              1. Predicted Price Change vs. Historic Average Daily Price Change: 
              [Insert analysis]
              2. Price Volatility Analysis: 
              [Insert analysis]
              3. 7-Day Trend and Slope Analysis: 
              [Insert analysis]

              Summary: 
              [Insert summary]

              Here is the expected Analysis Format to follow
              Analysis:
              Predicted Price Change ($6.14) is significantly above Historic Average Daily Change (-$0.85), with a substantial deviation of $6.99.
              Price Volatility (7.31) indicates moderate to high risk, but the predicted change is within this range.
              7-Day Trend is upward, but with a weak slope (0.12), making the next day's prediction less probable.
              Summary:
              Predicted price change is high-risk, but within volatility range. Upward trend is weak, making next day's prediction uncertain.


              Data: {data}
              
              Format your response as a structured, clear financial statistical analysis, limited to 5 lines maximum. Do not include repeated summaries.
              """
              )

  # Generate the advisory summary
  summary = prompt_template.format(data=data)
  print("I AM PRINTING SUMMARY", summary)
  response = llm.invoke(input=summary)   
  return response.content

  




@app.route('/', methods=['GET', 'POST'])

def index():
  result = None
  nextday = None
  nextsevenday = None
  filename = None
  stocksymbol = None
  str_nextsevenday =None
  str_nextday = None
  str_stock_mean_difference_price= None
  str_stock_volility_closing_price= None
  direction= None
  str_slope= None
  str_predicted_difference = None
  data_analysis =None
  adjusted_predictions =nextsevenday_adjusted =adjusted_plot_path=News_Trend=None
  PB_Ratio_Defination = stock_pb_ratio = stock_benchmark = stock_trend = stock_comment = pb_summary = \
  ROE_Defination = ticker_name = ticker_current = stock_roe_ratio = roe_stock_benchmark = roe_stock_trend = \
  roe_stock_comment = roe_summary = sma_defination = stock_sma_50 = sma_stock_benchmark = sma_stock_trend = \
  sma_stock_comment = sma_summary = revenue_change_defination = stock_revenue = revenue_stock_benchmark = \
  revenue_stock_trend = revenue_stock_comment = revenue_summary = debt_ratio_defination = stock_debt = \
  stock_benchmark_debt = stock_trend_debt = stock_comment_debt = summary_debt = ticker_selected = None


    

  if request.method == 'POST':
    if 'stocksymbol' in request.form:
      stocksymbol = request.form['stocksymbol']
      print("STOCKSYMBOL", stocksymbol)
      
      
      try:
          # Load the model
          BASE_DIR = os.path.dirname(os.path.abspath(__file__))
          MODELPATH = os.path.join(BASE_DIR, os.getenv("STOCKS_MODELPATH"))
          model_path = f"{MODELPATH}/{stocksymbol}.h5"
          stock_model_loaded = load_model(model_path)
          
          print("Stock model is loaded",stocksymbol)
          # Initialize PredictStocks with the stock name
          stockpredictor = PredictStocks(stocksymbol)
          print("Stock Class Initiated",stocksymbol,stockpredictor)
          
          # Use PredictStocks method to predict the next day
          nextday, nextsevenday,last15_predictions = stockpredictor.predict_stocks(stock_model_loaded)
          print("Next Day Predictions",nextday)
          str_nextday = str(nextday)
          
          print("Adjusted Calculations: ")
          news_influence = -0.8
          News_Trend ="Strong Negative"
          adjusted_predictions, nextsevenday_adjusted  = stockpredictor.news_adjusted(stocksymbol,last15_predictions,nextsevenday,news_influence)
          print("ADJUSTED ", adjusted_predictions)
          print("ADJUSTED NEXT", nextsevenday_adjusted )
          
          
          # Plot the graph and save it
          print("WE ARE PLOTING THE  PREDICTION GRAPH")
          plot_path = stockpredictor.plot_prediction(stocksymbol, nextsevenday,last15_predictions)
          
           # Plot ADJUSTED graph and save it
          print("WE ARE PLOTING THE ADJUSTED GRAPH")
          
          adjusted_plot_path = stockpredictor.plot_prediction_adjustment(stocksymbol, nextsevenday,last15_predictions,adjusted_predictions,nextsevenday_adjusted)

          #string the return
          str_nextsevenday = ', '.join(map(str, nextsevenday))

          print( str_nextsevenday )
          filename = os.path.basename(plot_path)
          #filename ="https://investiflex.com/stock_api/static/stock_prediction_plot.jpg"
          print("FILENAME", filename)
          
          stock_mean_difference_price, stock_volility_closing_price, direction,slope,predicted_difference = stockpredictor.prediction_eval(nextday, nextsevenday, stocksymbol)
          print(stock_mean_difference_price, stock_volility_closing_price, direction,slope)
          str_stock_mean_difference_price = str(round(stock_mean_difference_price,2))
          str_stock_volility_closing_price= str(round(stock_volility_closing_price,2))
          str_slope =str(round(slope,2))
          str_predicted_difference = str(round(predicted_difference,2))

          data = {
              "Next day's predicted price": nextday,
              "Predicted Price Change": predicted_difference,
              "Historic Average Daily Price Change": stock_mean_difference_price,
              "Price Volatility (Statistical Measure of Risk)": stock_volility_closing_price,
              "7-day prediction": nextsevenday,
              "7-Day Trend": direction,
              "Slope": round(slope, 2)
          }  
             
          data_analysis= llm_analysis(data)
          print(data_analysis)
          
          result = True  # Indicate success
          
          
      
       
          
      except Exception as e:
          # Handle any errors during prediction
          result = True  # Still render the page
          nextday, nextsevenday = "Error", "Error"
          filename = None
          
          

  # Render the template regardless of POST/GET
  return render_template_string (
        template,
        result=result,
        nextday=str_nextday,
        sevenday=str_nextsevenday,
        filename=filename,
        stocksymbol=stocksymbol,
        str_stock_mean_difference_price= str_stock_mean_difference_price,
        str_stock_volility_closing_price= str_stock_volility_closing_price,
        direction= direction,
        str_slope= str_slope,
        str_predicted_difference =str_predicted_difference,
        data_analysis = data_analysis,
        adjusted_predictions = adjusted_predictions,
        nextsevenday_adjusted= nextsevenday_adjusted,
        adjusted_plot_path=adjusted_plot_path,
        News_Trend=News_Trend,
        )
  

@app.route('/indicators', methods=['GET', 'POST'])
def route2():
  stock_symbol = request.args.get('stocksymbol')
  #print("Indicators Route")
  try:
      print(stock_symbol)
      results_indicators = calculate_indicators(stock_symbol)
        
      PB_Ratio_Defination, stock_pb_ratio, stock_benchmark, stock_trend, stock_comment, pb_summary, \
      ROE_Defination, ticker_name, ticker_current, stock_roe_ratio, roe_stock_benchmark, roe_stock_trend, \
      roe_stock_comment, roe_summary, sma_defination, stock_sma_50, sma_stock_benchmark, sma_stock_trend, \
      sma_stock_comment, sma_summary, revenue_change_defination, stock_revenue, revenue_stock_benchmark, \
      revenue_stock_trend, revenue_stock_comment, revenue_summary, debt_ratio_defination, stock_debt, \
      stock_benchmark_debt, stock_trend_debt, stock_comment_debt, summary_debt, ticker_selected = results_indicators.values()

          
      print(results_indicators)
        
  except Exception as e:
      print(e)
  
  from flask import render_template
  return render_template("indicators.html", PB_Ratio_Defination=PB_Ratio_Defination, stock_pb_ratio=stock_pb_ratio, 
                           stock_benchmark=stock_benchmark, stock_trend=stock_trend, stock_comment=stock_comment,
                           pb_summary=pb_summary, ROE_Defination=ROE_Defination, ticker_name=ticker_name, 
                           ticker_current=ticker_current, stock_roe_ratio=stock_roe_ratio, 
                           roe_stock_benchmark=roe_stock_benchmark, roe_stock_trend=roe_stock_trend, 
                           roe_stock_comment=roe_stock_comment, roe_summary=roe_summary, 
                           sma_defination=sma_defination, stock_sma_50=stock_sma_50, 
                           sma_stock_benchmark=sma_stock_benchmark, sma_stock_trend=sma_stock_trend, 
                           sma_stock_comment=sma_stock_comment, sma_summary=sma_summary, 
                           revenue_change_defination=revenue_change_defination, stock_revenue=stock_revenue, 
                           revenue_stock_benchmark=revenue_stock_benchmark, revenue_stock_trend=revenue_stock_trend, 
                           revenue_stock_comment=revenue_stock_comment, revenue_summary=revenue_summary, 
                           debt_ratio_defination=debt_ratio_defination, stock_debt=stock_debt, 
                           stock_benchmark_debt=stock_benchmark_debt, stock_trend_debt=stock_trend_debt, 
                           stock_comment_debt=stock_comment_debt, summary_debt=summary_debt, 
                           ticker_selected=ticker_selected )


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)
