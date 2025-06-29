#Imports
import pandas as pd
import yfinance as yf
import random
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import google.generativeai as genai
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_LLM_KEY= os.getenv("STOCKS_GOOGLE_KEY")
PB_MODEL_PATH = os.getenv("STOCKS_PB_MODEL_PATH")
ROE_MODEL_PATH = os.getenv("STOCKS_ROE_MODEL_PATH")
SMA_MODEL_PATH = os.getenv("STOCKS_SMA_MODEL_PATH")
REVENUE_MODEL_PATH = os.getenv("STOCKS_REVENUE_MODEL_PATH")
DEBT_MODEL_PATH = os.getenv("STOCKS_DEBT_MODEL_PATH")




#IMPORT GOOGLE KEYS
import google.generativeai as genai
os.environ['GOOGLE_API_KEY'] = GOOGLE_LLM_KEY

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model_google = genai.GenerativeModel('gemini-1.5-flash')


#IMPORT MODELS
# REPLACE PATH HERE

model_path_pb=PB_MODEL_PATH
model_saved_pb = GPT2LMHeadModel.from_pretrained(model_path_pb)
model_path_roe=ROE_MODEL_PATH
model_saved = GPT2LMHeadModel.from_pretrained(model_path_roe)
model_path_sma= SMA_MODEL_PATH
model_saved_sma = GPT2LMHeadModel.from_pretrained(model_path_sma)
model_path_revenue=REVENUE_MODEL_PATH
model_saved_revenue = GPT2LMHeadModel.from_pretrained(model_path_revenue)
model_path_debt=DEBT_MODEL_PATH
model_saved_debt = GPT2LMHeadModel.from_pretrained(model_path_debt)


# INITILIZE GP2 TRANSFORMER
# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token to GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token



#THESE ARE ALL THE CONFIGS

# @title 1
# ALL CONFIGS
def get_PB_ratio(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        Stockholder_equity= stock.balance_sheet.T['Stockholders Equity'].iloc[-4]
        #print(Stockholder_equity)
        '''
        if Stockholder_equity < 1e6:
            scale_factor = 1e3  # Stockholders' equity is in thousands
            #print("Stockholder's equity is in thousands, scaling by 1e3")
        elif Stockholder_equity < 1e9:
            scale_factor = 1e6  # Stockholders' equity is in millions
            #print("Stockholder's equity is in millions, scaling by 1e6")
        else:
            scale_factor = 1  # Already in correct scale
            #print("Stockholder's equity is already in the correct scale")

        # Scale the Stockholder's equity
        Stockholder_equity *= scale_factor
        '''

        market_price = stock.info['currentPrice']

        total_shares_outstanding = stock.info['sharesOutstanding']

        pb_ratio = market_price / (Stockholder_equity / total_shares_outstanding)
        #print(pb_ratio,"PB RATIO")
        return round(pb_ratio,4)

    except Exception as e:
        pb_ratio = "error"
        return "error"




config_pb_ratio = {
    "industry": "Technology",
    "benchmarks": {
        "Deep Discount": {"min": float('-inf'), "max": 0.005, "trend": "Outstanding Performance"},
        "Significant Discount": {"min": 0.005, "max": 0.05, "trend": "Outstanding Performance"},
        "Moderate Discount": {"min": 0.05, "max": 0.2, "trend": "Exceptional Profitability"},
        "Fair Value": {"min": 0.2, "max": 0.5, "trend": "Adequate Profitability"},
        "Slight Premium": {"min": 0.5, "max": 1.0, "trend": "Low Profitability"},
        "Moderate Premium": {"min": 1.0, "max": 2.0, "trend": "Break Even"},
        "Significant Premium": {"min": 2.0, "max": 5.0, "trend": "Moderate Losses"},
        "High Premium": {"min": 5.0, "max": 10.0, "trend": "Substantial Losses"},
        "Extremely High Premium": {"min": 10.0, "max": float('inf'), "trend": "Significant Losses"}
    }
}



def predict_pb_ratio(pb_ratio):
    print("benchmarks")
    benchmarks = config_pb_ratio['benchmarks']

    for category, range in benchmarks.items():
        if range["min"] <= pb_ratio <= range["max"]:
            return {
                "category": category,
                "trend": range["trend"]
            }

    return {
        "category": "Unknown Category",
        "trend": "Unknown Trend"
    }



###SMA#####

# Function to get ROE change and current ROE
def get_SMA_50(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        data = stock.history(period="1y")
        sma_50_last= data['Close'].rolling(window=200).mean()[-1]
        return round(sma_50_last,2)
    except Exception as e:
        sma_50_last = "error"
        return "error"



config_sma_50 = {
    "industry": "Technology",
    "benchmarks": {
        "Extreme Undervaluation": {"min": float('-inf'), "max": 10, "trend": "Outstanding Performance"},
        "Significant Surplus": {"min": 10, "max": 30, "trend": "Outstanding Performance"},
        "Very Low Debt": {"min": 30, "max": 40, "trend": "Exceptional Profitability"},
        "Low Debt": {"min": 40, "max": 60, "trend": "Adequate Profitability"},
        "Moderate Debt": {"min": 60, "max": 80, "trend": "Low Profitability"},
        "Balanced Capital Structure": {"min": 80, "max": 90, "trend": "Break Even"},
        "Increasing Leverage": {"min": 90, "max": 120, "trend": "Moderate Losses"},
        "Highly Leveraged": {"min": 120, "max": 160, "trend": "Substantial Losses"},
        "Severe Debt Load": {"min": 160, "max": float('inf'), "trend": "Significant Losses"}
    }
}



def predict_sma_50_trend(sma_50_value):
    industry = config_sma_50['industry']
    benchmarks = config_sma_50['benchmarks']

    for category, range in benchmarks.items():
        if range["min"] <= sma_50_value <= range["max"]:
            return {
                "category": category,
                "trend": range["trend"]
            }

    return {
        "category": "Unknown Category",
        "trend": "Unknown Trend"
    }




######GET ROE

# Function to get ROE change and current ROE
def get_roe(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        #print(stock)

        # ROE LAST QUARTER
        Net_Income_current = stock.financials.T['Net Income'].iloc[0]
        #print(Net_Income_current)

        Stockholders_Equity_current = stock.balance_sheet.T['Stockholders Equity'].iloc[0]
        ROE_current = Net_Income_current / Stockholders_Equity_current

        Net_Income_last = stock.financials.T['Net Income'].iloc[-4]
        Stockholders_Equity_last = stock.balance_sheet.T['Stockholders Equity'].iloc[-4]
        ROE_last = Net_Income_last / Stockholders_Equity_last

        # Check for division by zero in ROE_last
        if ROE_last == 0:
            return "error", "error"

        # Calculate ROE change as a percentage
        ROE_change = ((ROE_current - ROE_last) / ROE_last) * 100

        # Check for NaN values
        if np.isnan(ROE_current) or np.isnan(ROE_change):
            return "error", "error"

        return round(ROE_change, 2), round(ROE_current, 2)

    except Exception as e:
        #print(f"Error: {e}")
        return "error", "error"


# Configuration for ROE change predictions


config_roe_change = {
    "industry": "Technology",
    "benchmarks": {
        "Extreme Debt and Risk": {"min": float('-inf'), "max": -50, "trend": "Severe Losses"},
        "Concerning Debt": {"min": -50, "max": 0, "trend": "Moderate Losses"},
        "Very Low Debt": {"min": 0, "max": 20, "trend": "Break Even"},
        "Low Debt": {"min": 20, "max": 50, "trend": "Adequate Profitability"},
        "Moderate Debt": {"min": 50, "max": 70, "trend": "Exceptional Profitability"},
        "Balanced Capital Structure": {"min": 70, "max": 100, "trend": "Outstanding Performance"},
        "Increasing Leverage": {"min": 100, "max": 150, "trend": "Outstanding Performance"},
        "Highly Leveraged": {"min": 150, "max": float('inf'), "trend": "Significant Surplus"}
    }
}

# Function to predict the trend based on ROE change
def predict_pb_ratio(ROE_change):
    benchmarks = config_roe_change['benchmarks']

    for category, range in benchmarks.items():
        if range["min"] <= ROE_change <= range["max"]:
            return {
                "category": category,
                "trend": range["trend"]
            }

    return {
        "category": "Unknown Category",
        "trend": "Unknown Trend"
    }




#REVENU CHANGE CONFIG

def get_operating_income(ticker_selected):
    try:
        stock = yf.Ticker(ticker_selected)
        #print(stock)
        df = stock.quarterly_income_stmt.T[['Total Revenue']].dropna()
        #print(df.shape, df.columns)
        # Calculate Quarterly (QoQ) Change
        df['QoQ Change'] = df['Total Revenue'].pct_change(periods=1)



        # Convert percentage changes to decimal format
        df['QoQ Change (%)'] = df['QoQ Change'] * 100

        #print(df[['QoQ Change (%)']].mean())
        data = round(df[['QoQ Change (%)']].mean(),3)[0]
        return data
    except Exception as e:
        data = "error"
        return data



config_revenue_growth = {
    "industry": "General",
    "benchmarks": {
        "Severe Revenue Decrease ": {"min": -float('inf'), "max": -20, "trend": "Significant Losses"},
        "Highly Leveraged": {"min": -20, "max": -15, "trend": "Substantial Losses"},
        "Increasing Leverage": {"min": -15, "max": -10, "trend": "Moderate Losses"},
        "Balanced Capital Structure": {"min": -10, "max": 0, "trend": "Break Even"},
        "Slow Growth": {"min": 0, "max": 5, "trend": "Adequate Profitability"},
        "Steady Growth": {"min": 5, "max": 10, "trend": "Exceptional Profitability"},
        "Significant Profits and Growth": {"min": 10, "max": 20, "trend": "Outstanding Performance"},
        "Extreme Undervaluation": {"min": 20, "max": float('inf'), "trend": "Outstanding Performance"}
    }
}



def predict_operating_income(operating_income_change):
    benchmarks = config_revenue_growth ['benchmarks']

    for category, range in benchmarks.items():
        if range["min"] <= operating_income_change <= range["max"]:
            return {
                "category": category,
                "trend": range["trend"]
            }

    return {
        "category": "Unknown Category",
        "trend": "Unknown Trend"
    }



#DEBTH RATIO

#get Ratio , Benchmark , Signal
def get_debt_ratio(ticker_selected):
    try:
        stock = yf.Ticker(ticker_selected)
        debt_ratio= stock.balance_sheet.T['Total Debt'][-4]/stock.balance_sheet.T['Stockholders Equity'][-4]
        return round(debt_ratio,2)
    except Exception as e:
        debt_ratio = "error"
        return "error"


config_debt = {
  "industry": "Technology",
  "benchmarks": {
    "Significant Surplus": {"min": float('-inf'), "max": -0.25, "trend": "Outstanding Performance"},
    "Very Low Debt": {"min": -0.25, "max": 0, "trend": "Exceptional Profitability"},
    "Debt-Free or Minimal Debt": {"min": 0, "max": 0.25, "trend": "Strong Profitability"},
    "Low Debt": {"min": 0.25, "max": 0.5, "trend": "Adequate Profitability"},
    "Moderate Debt": {"min": 0.5, "max": 0.75, "trend": "Low Profitability"},
    "Balanced Capital Structure": {"min": 0.75, "max": 1.25, "trend": "Break Even"},
    "Increasing Leverage": {"min": 1.25, "max": 1.75, "trend": "Moderate Losses"},
    "Highly Leveraged": {"min": 1.75, "max": 2.5, "trend": "Substantial Losses"},
    "Severe Debt Load": {"min": 2.5, "max": float('inf'), "trend": "Significant Losses"}
  }
}


def predict_debt_ratio(debt_ratio):
    benchmarks = config_debt['benchmarks']

    for category, benchmark in benchmarks.items():
        if benchmark["min"] <= debt_ratio <= benchmark["max"]:
            return {
                "category": category,
                "trend": benchmark["trend"]
            }

    return {
        "category": "Unknown Category",
        "trend": "Unknown Trend"
    }




#TRANSFORMER AGENTS

#AGENT 1 PB RATIO

def pb_ratio_model(ticker_selected):
  print("INDICATOR - Price to Book Ratio")
  try:
    pb_ratio= get_PB_ratio(ticker_selected)
  
    prediction =predict_pb_ratio(pb_ratio)
    #print(prediction)
    benchmark =prediction['category']
    trend =  prediction['trend']

    

    test_input_pb = (
        f"Indicator: {str(float(pb_ratio))}",
        f"benchmark: {str(benchmark)}",
        f"trend: {str(trend)}",
        "comment:"
    )

      #print(test_input_pb)

        # Check for invalid categories or trends
    test_input1 = test_input_pb
    
    if test_input1[0].split(" ")[1] == "nan" or test_input1[1].split(" ")[1] == "Unknown Category" or \
      test_input1[2].split(" ")[1] == "Unknown Trend":
        print("Try another Stock, data value not found")
    else:
        # Concatenate the input list into a single string for tokenization
        input_text = " ".join(test_input1)
        #print(f"Concatenated Input: {input_text}")  # Debugging: Check concatenated input

        # Set padding token before encoding
        tokenizer.pad_token = tokenizer.eos_token

        # Encode the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Debugging: Print input to the model
        #print(f"Input to the model: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

        # Generate text using the loaded model
        generated_text_samples = model_saved_pb.generate(
            input_ids,
            max_length=150,  # Increase length if needed
            num_return_sequences=1,
            attention_mask=attention_mask,  # Pass the attention mask
            pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
            temperature=.9,  # Lower temperature for more deterministic output
            do_sample=True,  # Enable sampling
            top_k=0,  # Optional: Limits the number of tokens to sample from 0 allows all options token 50 allows just 50 high prob tokens
            top_p=0.95  # Optional: Nucleus sampling; keeps the top 95% probability mass
        )

        PB_Ratio_Defination= "The Price-to-Book (P/B) Ratio is calculated by dividing a company's market price per share by its book \
        value per share (Stockholders' Equity divided by total shares outstanding). It measures how much \
        investors are willing to pay for each dollar of a company's assets, helping assess whether a stock is undervalued or overvalued."

        # Decode and print the generated text
        '''
        for i, sample in enumerate(generated_text_samples):
            print(f"Generated Text {i+1}: {tokenizer.decode(sample, skip_special_tokens=True)}")
        '''
        response = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
        ticker_current = ticker_selected
        ticker_name= yf.Ticker(ticker_current)
        company_info = ticker_name.info
        ticker_name= company_info['longName']
        print (ticker_current,ticker_name)
        #
        stock_pb_ratio= response.split("icator: ")[1].split(",")[0]
        stock_benchmark = response.split("benchmark: ")[1].split(",")[0]
        stock_trend = response.split("trend: ")[1].split(",")[0]
        stock_comment = response.split("comment: ")[1]

        #print (f'Indicator is PB Ratio Defination : {PB_Ratio_Defination}\n ')
        print("***")
        print(f'PB Ratio is : {stock_pb_ratio.split(" ")[0] }\n')

        print(f'Benchmark is : {stock_benchmark.split("trend:")[0]}\n')
        print(f'Trend is : {stock_trend.split("comment:")[0] }\n')
        print(f'Comment Generated : {stock_comment}\n')

        g_prompt =(f'give financial analysis of the following . this is what we are calculating{PB_Ratio_Defination}.\
        Explain the PB ratio of a stock {ticker_name}, ticker symbol {ticker_current} ,\
        the PB ratio is {stock_pb_ratio }, the benchmark for this value is {stock_benchmark}\
        the trend for this value is {stock_trend} and a short commment is {stock_comment}, please explain this in 2 lines\
        Explain if the stock is a good investment.')

        summary= model_google.generate_content(g_prompt).text
        

        print(f'Summary : {summary}')
        
        stock_benchmark1= str(stock_benchmark.split("trend:")[0])
        stock_trend1= str(stock_trend.split("comment:")[0]) 
        print("HEHEHEHEHEHE111")
        return PB_Ratio_Defination, ticker_name,ticker_current, stock_pb_ratio ,stock_benchmark1,\
        stock_trend1, stock_comment, summary

  except Exception as e:
    print(e)
    print("Please try another stock, no data found")
    print("\n")
    print("-----")
    
    


#MODEl 2 AGENt 2 ROE return to equity

# @title
def roe_model(ticker_selected):
  print("INDICATOR - Return of Equity")

  try:
      # Get ROE change and current ROE
      ROE_change, ROE_current = get_roe(ticker_selected)

      # If there's an error, skip further execution
      if ROE_change == "error" or ROE_current == "error":
          print("try another stock, no data found")
      else:
          # Predict based on ROE_change
          prediction = predict_pb_ratio(ROE_change)
          #print(prediction)
          benchmark = prediction['category']
          trend = prediction['trend']

          #print(f"Benchmark: {benchmark}, Trend: {trend}, ROE Change: {ROE_change}, ROE Current: {ROE_current}")

          # Create input for further analysis
          test_input_roe = (
              f"Indicator: {str(ROE_change)}%",
              f"Benchmark: {str(benchmark)}",
              f"Trend: {str(trend)}",
              "Comment:"
          )

          #print(test_input1)


          test_input1 = test_input_roe
          # Check for invalid categories or trends
          if test_input1[0].split(" ")[1] == "nan" or test_input1[1].split(" ")[1] == "Unknown Category" or \
            test_input1[2].split(" ")[1] == "Unknown Trend":
              print("Try another Stock, datavalue not found")
          else:
              # Concatenate the input list into a single string for tokenization
              input_text = " ".join(test_input1)
              #print(f"Concatenated Input: {input_text}")  # Debugging: Check concatenated input

              # Set padding token before encoding
              tokenizer.pad_token = tokenizer.eos_token

              # Encode the input text
              input_ids = tokenizer.encode(input_text, return_tensors="pt")

              # Generate attention mask
              attention_mask = (input_ids != tokenizer.pad_token_id).long()

              # Debugging: Print input to the model
              #print(f"Input to the model: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

              # Generate text using the loaded model
              generated_text_samples = model_saved.generate(
                  input_ids,
                  max_length=150,  # Increase length if needed
                  num_return_sequences=1,
                  attention_mask=attention_mask,  # Pass the attention mask
                  pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
                  temperature=.9,  # Lower temperature for more deterministic output
                  do_sample=True,  # Enable sampling
                  top_k=0,  # Optional: Limits the number of tokens to sample from 0 allows all options token 50 allows just 50 high prob tokens
                  top_p=0.95  # Optional: Nucleus sampling; keeps the top 95% probability mass
              )

              response = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)


              ROE_Defination= "ROE change measures the percentage difference between the return on equity from one\
              period to the next, indicating how efficiently a company generates profit relative to shareholders' equity over time.\
              A positive ROE change suggests improved profitability, while a negative change signals potential declines in financial performance or efficiency."



              # Get current ticker
              ticker_current = ticker_selected
              ticker_name = yf.Ticker(ticker_current)
              company_info = ticker_name.info
              ticker_name = company_info['longName']
              print(ticker_current, ticker_name)

              # Extracting values from the response
              stock_roe_ratio = response.split("Indicator: ")[1].split(" ")[0]
              stock_benchmark = response.split("Benchmark: ")[1].split(" Trend")[0]
              stock_trend = response.split("Trend: ")[1].split(" Comment")[0]
              stock_comment = response.split("Comment: ")[1]

              # Define PB Ratio Definition (you may want to provide the actual definition text here)
              #PB_Ratio_Defination = "The PB ratio compares a company's market price to its book value."

              # Printing results
              #print(f'Indicator is ROE Definition: {ROE_Defination}\n')
              print("***")
              print(f'ROE % is: {stock_roe_ratio }\n')
              print(f'Benchmark is: {stock_benchmark}\n')
              print(f'Trend is: {stock_trend}\n')
              print(f'Comment Generated: {stock_comment}\n')

              g_prompt =(f'give financial analysis of the following . this is what we are calculating{ROE_Defination}.\
              Explain the Return to Equity of a stock {ticker_name}, ticker symbol {ticker_current} ,\
              the Return to Equity is {stock_roe_ratio }, the benchmark for this value is {stock_benchmark}\
              the trend for this value is {stock_trend} and a short commment is {stock_comment}, please explain this in 2 lines\
              Explain if the stock is a good investment')



              summary= model_google.generate_content(g_prompt).text
              #print(f'Summary of the indicator : {summary}')
              print(f'Summary : {summary}')
                

        
              return ROE_Defination, ticker_name, ticker_current, stock_roe_ratio ,stock_benchmark,\
                 stock_trend, stock_comment, summary
            

  except Exception as e:
      print("Try another stock, no data found")
      print("\n")
      print("-----")
      
      

#AGENT 3  Simple moving average
# @title
def sma_model(ticker_selected):
  print("INDICATOR - Simple Moving Average")
  try:
      # Get ROE change and current ROE
      sma_50 = get_SMA_50(ticker_selected)
      #print(sma_50)

      # If there's an error, skip further execution
      if sma_50 == "error":
          print("try another stock, no data found")
      else:
          # Predict based on ROE_change
          prediction = predict_sma_50_trend(sma_50)
          #print(prediction)
          benchmark = prediction['category']
          trend = prediction['trend']

          #print(f"Benchmark: {benchmark}, Trend: {trend}, ROE Change: {sma_50}")

          # Create input for further analysis
          test_input_sma = (
              f"Indicator: {str(sma_50)}",
              f"Benchmark: {str(benchmark)}",
              f"Trend: {str(trend)}",
              "Comment:"
          )


          test_input1 = test_input_sma
          # Extract actual values
          indicator_value = test_input1[0].split(": ")[1].strip().lower()
          benchmark_value = test_input1[1].split(": ")[1].strip().lower()
          trend_value = test_input1[2].split(": ")[1].strip().lower()

          if indicator_value in ["nan", "none", ""] or benchmark_value in ["unknown category", "none", ""] or trend_value in ["unknown trend", "none", ""]:
              print("Try another stock, data value not found")

          else:

              # Concatenate the input list into a single string for tokenization
              input_text = " ".join(test_input1)
              #print(f"Concatenated Input: {input_text}")  # Debugging: Check concatenated input

              # Set padding token before encoding
              tokenizer.pad_token = tokenizer.eos_token

              # Encode the input text
              input_ids = tokenizer.encode(input_text, return_tensors="pt")

              # Generate attention mask
              attention_mask = (input_ids != tokenizer.pad_token_id).long()

              # Debugging: Print input to the model
              #print(f"Input to the model: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

              # Generate text using the loaded model
              generated_text_samples = model_saved_sma.generate(
                  input_ids,
                  max_length=150,  # Increase length if needed
                  num_return_sequences=1,
                  attention_mask=attention_mask,  # Pass the attention mask
                  pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
                  temperature=.9,  # Lower temperature for more deterministic output
                  do_sample=True,  # Enable sampling
                  top_k=0,  # Optional: Limits the number of tokens to sample from 0 allows all options token 50 allows just 50 high prob tokens
                  top_p=0.95  # Optional: Nucleus sampling; keeps the top 95% probability mass
              )

              response = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
              #print(response)

              sma_defination = "The SMA 200-day is interpreted as a long-term trend indicator; \
              if the stock price is above the SMA 200, it suggests a bullish trend, while a price below the SMA 200 \
              indicates a bearish trend."



              # Get current ticker
              ticker_current = ticker_selected
              ticker_name = yf.Ticker(ticker_current)
              company_info = ticker_name.info
              ticker_name = company_info['longName']
              print(ticker_current, ticker_name)

              # Extracting values from the response
              stock_sma_50 = response.split("Indicator: ")[1].split(" ")[0]
              stock_benchmark = response.split("Benchmark: ")[1].split(" Trend")[0]
              stock_trend = response.split("Trend: ")[1].split(" Comment")[0]
              stock_comment = response.split("Comment: ")[1]

              # Define PB Ratio Definition (you may want to provide the actual definition text here)
              #PB_Ratio_Defination = "The PB ratio compares a company's market price to its book value."

              # Printing results
              #print(f'Indicator is ROE Definition: {ROE_Defination}\n')
              print(f'Simple moving Average 200 : {stock_sma_50 }\n')
              print(f'Benchmark is: {stock_benchmark}\n')
              print(f'Trend is: {stock_trend}\n')
              print(f'Comment Generated: {stock_comment}\n')

              price = yf.Ticker(ticker_selected)  # Replace with desired stock ticker
              data = price.info

              current_price = data['currentPrice']

              g_prompt =(f'give financial analysis of the following . this is what we are calculating{sma_defination}.\
              Explain the simple moving avaerage 200 days of a stock {ticker_name}, ticker symbol {ticker_current} ,\
              the current price to compare is {current_price} include this in your response and compare too average,\
              the sma 200 is {stock_sma_50 }, the benchmark for this value is {stock_benchmark}\
              the trend for this value is {stock_trend} and a short commment is {stock_comment}, please explain this in 2 lines\
              Explain if the stock is a good investment.')



              summary= model_google.generate_content(g_prompt).text
              #print(f'Summary of the indicator : {summary}')
              print(f'Summary : {summary}')
            
              return sma_defination, ticker_name, ticker_current, stock_sma_50 ,stock_benchmark,\
              stock_trend, stock_comment, summary

  except Exception as e:
    print("Try another stock, no data found")
    print("\n")
    print("-----")
    


#AGENT 4 Revenue Model

# @title
def revenue_model(ticker_selected):
  print("INDICATOR - REVENUE CHANGE OVER QUARTER")

  try:
    # Get ROE change and current ROE
    operating_income_change = get_operating_income(ticker_selected)

    # If there's an error, skip further execution
    if operating_income_change == "error":
      print("try another stock, no data found,OINCOME")
    else:
      # Predict based on ROE_change
      prediction = predict_operating_income(operating_income_change)
      #print(prediction)
      benchmark = prediction['category']
      trend = prediction['trend']

      #print(f"Benchmark: {benchmark}, Trend: {trend}, ROE Change: {operating_income_change}")

      # Create input for further analysis
      test_input_revenue = (
          f"Indicator: {str(operating_income_change)}%",
          f"Benchmark: {str(benchmark)}",
          f"Trend: {str(trend)}",
          "Comment:"
      )

      #print(test_input_revenue,"INPUT")


      test_input1 = test_input_revenue
      # Extract actual values
      indicator_value = test_input1[0].split(": ")[1].strip().lower()
      benchmark_value = test_input1[1].split(": ")[1].strip().lower()
      trend_value = test_input1[2].split(": ")[1].strip().lower()

      if indicator_value in ["nan", "none", ""] or benchmark_value in ["unknown category", "none", ""] or trend_value in ["unknown trend", "none", ""]:
          print("Try another stock, data value not found")

      else:

          # Concatenate the input list into a single string for tokenization
          input_text = " ".join(test_input1)
          #print(f"Concatenated Input: {input_text}")  # Debugging: Check concatenated input

          # Set padding token before encoding
          tokenizer.pad_token = tokenizer.eos_token

          # Encode the input text
          input_ids = tokenizer.encode(input_text, return_tensors="pt")

          # Generate attention mask
          attention_mask = (input_ids != tokenizer.pad_token_id).long()

          # Debugging: Print input to the model
          #print(f"Input to the model: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

          # Generate text using the loaded model
          generated_text_samples = model_saved_revenue.generate(
              input_ids,
              max_length=150,  # Increase length if needed
              num_return_sequences=1,
              attention_mask=attention_mask,  # Pass the attention mask
              pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
              temperature=.9,  # Lower temperature for more deterministic output
              do_sample=True,  # Enable sampling
              top_k=0,  # Optional: Limits the number of tokens to sample from 0 allows all options token 50 allows just 50 high prob tokens
              top_p=0.95  # Optional: Nucleus sampling; keeps the top 95% probability mass
          )

          response = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
          #print(response)

          revenue_change_defination =  "Revenue change % over a quarter measures the percentage \
          difference in a company's revenue between the current quarter and the previous quarter.\
           It helps assess growth or decline in sales performance over a short-term period."



          # Get current ticker
          ticker_current = ticker_selected
          ticker_name = yf.Ticker(ticker_current)
          company_info = ticker_name.info
          ticker_name = company_info['longName']
          print(ticker_current, ticker_name)

          # Extracting values from the response
          stock_revenue = operating_income_change
          stock_benchmark = response.split("Benchmark: ")[1].split(" Trend")[0]
          stock_trend = response.split("Trend: ")[1].split(" Comment")[0]
          stock_comment = response.split("Comment: ")[1]



          # Printing results
          #print(f'Indicator is ROE Definition: {ROE_Defination}\n')
          print(f'Revenue Change : {stock_revenue }\n')
          print(f'Benchmark is: {stock_benchmark}\n')
          print(f'Trend is: {stock_trend}\n')
          print(f'Comment Generated: {stock_comment}\n')


          g_prompt =(f'give financial analysis of the following . this is what we are calculating{revenue_change_defination}.\
          Explain the revene change % over quarters {ticker_name}, ticker symbol {ticker_current} ,\
          the revenue change is {stock_revenue }, the benchmark for this value is {stock_benchmark}\
          the trend for this value is {stock_trend} and a short commment is {stock_comment}, please explain this in 2 lines\
          Explain if the stock is a good investment.')



          summary= model_google.generate_content(g_prompt).text
          #print(f'Summary of the indicator : {summary}')
          print(f'Summary : {summary}')
            
          return revenue_change_defination, ticker_name, ticker_current, stock_revenue ,stock_benchmark,\
          stock_trend, stock_comment, summary
            
        

  except Exception as e:
    print("Try another stock, no data found")
    print("\n")
    print("-----")
    


#Agent Revenue 5  debt ratio 
# @title
def debth_model(ticker_selected):
  print("INDICATOR - DEBT RATIO ")

  try:
    debt_ratio= get_debt_ratio(ticker_selected)
    prediction =predict_debt_ratio(debt_ratio)
    benchmark =prediction['category']
    trend =  prediction['trend']

    #print(benchmark,trend,debt_ratio)
              # Get current ticker
    ticker_current = ticker_selected
    ticker_name = yf.Ticker(ticker_current)
    company_info = ticker_name.info
    ticker_name = company_info['longName']
    print(ticker_current, ticker_name)


    test_input_debt = (
        f"Indicator: {str(debt_ratio)}",
        f"benchmark: {str(benchmark)}",
        f"trend: {str(trend)}",
        "Comment:"
    )
    #print(test_input_debt,"INPUT")


    test_input = test_input_debt

        #get Comment from Transformer
    if test_input[0].split(" ")[1]=="nan" or test_input[1].split(" ")[1]=="Unknown Category" or \
        test_input[2].split(" ")[1]=="Unknown Trend":
        print("Try another Stock, datavalue not found")
    else:
      # Encode the input text
      input_ids = tokenizer.encode(test_input, return_tensors="pt")

      # Add padding token to GPT-2 tokenizer
      tokenizer.pad_token = tokenizer.eos_token

      # Generate attention mask
      attention_mask = (input_ids != tokenizer.pad_token_id).long()

      generated_text_samples = model_saved_debt.generate(
        input_ids,
        max_length=150,  # Increase length if needed
        num_return_sequences=1,
        attention_mask=attention_mask,  # Pass the attention mask
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
        temperature=.9,  # Lower temperature for more deterministic output
        do_sample=True,  # Enable sampling
        top_k=0,  # Optional: Limits the number of tokens to sample from 0 allows all options token 50 allows just 50 hight prob tokens
        #top_p=0.95  # Optional: Nucleus sampling; keeps the top 95% probability mass. 1 will allow all probabilties .9 will inclue all that adds to .9
        )

      response = tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
      #print("RESPONSE",response)

      debt_ratio_defination = "The debt-to-equity ratio calculates a company's financial leverage \
      by dividing Total Debt by Stockholders' Equity,indicating financial health and risk.\
      A lower ratio suggests conservative financing and low risk,while a higher ratio indicates \
      aggressive financing and increased risk of financial distress"



      # Get current ticker
      ticker_current = ticker_selected
      ticker_name= yf.Ticker(ticker_current)
      company_info = ticker_name.info
      ticker_name= company_info['longName']
      #print (ticker_current,ticker_name)

      # Extracting values from the response
      stock_debt = response.split("icator: ")[1].split(",")[0]

      stock_benchmark = response.split("benchmark: ")[1].split(" trend")[0]

      stock_trend = response.split("trend: ")[1].split(" comment")[0]

      stock_comment = response.split("comment: ")[1]



      # Printing results
      #print(f'Indicator is ROE Definition: {ROE_Defination}\n')
      print(f'Debt Ratio : {stock_debt }\n')
      print(f'Benchmark is: {stock_benchmark}\n')
      print(f'Trend is: {stock_trend}\n')
      print(f'Comment Generated: {stock_comment}\n')


      g_prompt =(f'give financial analysis of the following . this is what we are calculating{debt_ratio_defination}.\
      Explain the revene change % over quarters {ticker_name}, ticker symbol {ticker_current} ,\
      the revenue change is {stock_debt }, the benchmark for this value is {stock_benchmark}\
      the trend for this value is {stock_trend} and a short commment is {stock_comment}, please explain this in 2 lines\
      Explain if the stock is a good investment.')



      summary= model_google.generate_content(g_prompt).text
      #print(f'Summary of the indicator : {summary}')
      print(f'Summary : {summary}')
    
      return debt_ratio_defination, ticker_name, ticker_current, stock_debt ,stock_benchmark,\
      stock_trend, stock_comment, summary

  except Exception as e:
    print("Try another stock, no data found")
    print("\n")
    print("-----")
    
    






