from flask import Flask, request, render_template_string
from flask import url_for
import threading
import Indicators
from Indicators import *



#HTML TEMPLATE

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">

    <style>
    body {
    background-color:black;
    font-family: 'Roboto', sans-serif;

    }
    
    h1 {
    font-weight:500;
    color:white;
  
    
    }
    
    h3 {
    color: #8175ff;
    font-weight:500;
    }
    
    .pagetitle {text-align: center; color: #8175ff; }
    
    .subtitle {text-align: center;}
    
    
    p {
    font-size : 1em;
    }
    
    .divindicator {
        width: 18%;
        color: white;
        margin-right: -13.5% !important;
        border-right: 1px solid rgba(255, 255, 255, 0.20);
        box-sizing: border-box;
        flex: 0 0 auto;
        margin: 0;
        padding: 0;
        }


   .linedevider{
   background-color:rgba(255, 255, 255, 0.20);
   height:1px;
   width:100%;
   }
    
    .container {
       display: flex;
    justify-content: space-between;
    width: 100%;
    overflow: hidden;
    }
    
    
    .bignumber {
    font-size : 4em;
    font-weight:100;
    margin : 0px!important;
    color: #00c3ff;
    }
    
    .textsummary {
    font-size : 19px;
    font-weight :200;
    line-height:130%;
    margin-right:10%;
    }
    
     .textsummary1 {
    font-size : 13px;
    font-weight :200;
    line-height:130%;
    margin-right:10%;
    }
    
    hr {
    width: 100%; /* Adjust width */
    opacity: 0.2; /* 40% opacity */
    border: none; /* Remove default border */
    height: 1px; /* Set height */
    background-color: white; /* Change color if needed */
    }
    
    .footer{
    padding-bottom :20%;
    }
    
    
    .row_container {
         display: flex;
        flex-wrap: nowrap;
        justify-content: flex-start;
        margin-right: -1% !important;
        margin-left: 1%;

    }''
    
    </style>
    
    
</head>



<body>



    <h1 class= "pagetitle">{{ title }}</h1>
    
    <h3 class= "subtitle" >Select a Stock</h2>
    
    <form class= "subtitle" action="/" method="POST">
            <select name="stock",  required>
                <option value="AAPL">AAPL</option>
                <option value="MSFT">MSFT</option>
                <option value="TSLA">TSLA</option>
            </select>
            <button type="submit">Submit</button>
    </form>
    
    
    
   
    



{% if ticker_selected %}

 <h3 class= "subtitle">{{ subtitle }}</h3>
  <div class ="linedevider"> </div>  

<div class="container">    


<div class="row_container">


        <div class="divindicator">
            
            <p> Insight #1</p>
            <h3> Price/Book Ratio </h3>

            <p class="bignumber"> {{stock_pb_ratio}}  </p>
            <p class="textsummary"> <b> Benchmark Value: <br> </b> {{stock_benchmark}} <br> <b>Trend :<br> </b>{{stock_trend}}</p>
            <hr>


            <p class="textsummary" ><b> Agent Insight: </b> <br>  {{ pb_summary }}</p>

            <p class="textsummary"> <b> Agent Statistical Analysis:<br> </b> {{stock_comment}}  </p>

            <p class="textsummary1" > <b> Defination: </b> <br> {{PB_Ratio_Defination}}</p>

            <p class ="footer"></p>
          
        </div>

<--! ROE INSIGHTS MODEL 2 --//> 


        <div class="divindicator">
           
            <p> Insight #2</p>
            <h3> Return of Equity </h3>

            <p class="bignumber"> {{stock_roe_ratio}}  </p>
            <p class="textsummary"> <b> Benchmark Value: <br> </b> {{roe_stock_benchmark}} <br> <b>Trend :<br> </b> {{roe_stock_trend}}</p>
            <hr>


            <p class="textsummary" ><b> Agent Insight : </b> <br>{{roe_summary}}</p>

            <p class="textsummary"> <b> Agent Statistical Analysis :<br> </b>{{roe_stock_comment}}</p>

            <p class="textsummary1" > <b> Defination: </b> <br> {{ROE_Defination}}</p>

            <p class ="footer"></p>
         
        </div>
        
        

<--! SMA INSIGHTS MODEL 3 --//> 


     <div class="divindicator">
            
            <p> Insight #3</p>
            <h3> Simple Moving Average </h3>

            <p class="bignumber"> {{stock_sma_50}}  </p>
            <p class="textsummary"> <b> Benchmark Value: <br> </b> {{sma_stock_benchmark}} <br> <b>Trend :<br> </b> {{sma_stock_trend}}</p>
            <hr>


            <p class="textsummary" ><b> Agent Insight : </b> <br>{{sma_summary}}</p>

            <p class="textsummary"> <b> Agent Statistical Analysis :<br> </b>{{sma_stock_comment}}</p>

            <p class="textsummary1" > <b> Defination: </b> <br> {{sma_defination}}</p>

            <p class ="footer"></p>
         
        </div>
        
        
<--! REVENUE INSIGHTS MODEL 4 --//> 


         <div class="divindicator">
            
            <p> Insight #4</p>
            <h3> Revenue Change </h3>

            <p class="bignumber"> {{stock_revenue}}  </p>
            <p class="textsummary"> <b> Benchmark Value: <br> </b> {{revenue_stock_benchmark}} <br> <b>Trend :<br> </b> {{revenue_stock_trend}}</p>
            <hr>


            <p class="textsummary" ><b> Agent Insight : </b> <br>{{revenue_summary}}</p>

            <p class="textsummary"> <b> Agent Statistical Analysis :<br> </b>{{revenue_stock_comment}}</p>

            <p class="textsummary1" > <b> Defination: </b> <br> {{revenue_change_defination}}</p>

            <p class ="footer"></p>
         
        </div>    
        
        

<--! ROE INSIGHTS MODEL 5 --//>  


         <div class="divindicator">
           
            <p> Insight #5</p>
            <h3> Debt Ratio </h3>

            <p class="bignumber"> {{stock_debt}}  </p>
            <p class="textsummary"> <b> Benchmark Value: <br> </b> {{stock_benchmark_debt}} <br> <b>Trend :<br> </b> {{stock_trend_debt}}</p>
            <hr>


            <p class="textsummary" ><b> Agent Insight : </b> <br>{{summary_debt}}</p>

            <p class="textsummary"> <b> Agent Statistical Analysis :<br> </b>{{stock_comment_debt}}</p>

            <p class="textsummary1" > <b> Defination: </b> <br> {{debt_ratio_defination}}</p>

            <p class ="footer"></p>
         
        </div>

    

</div>
</div>  


{% endif %}

</body>
</html>
'''

PB_Ratio_Defination = stock_pb_ratio = stock_benchmark = stock_trend = stock_comment = pb_summary = \
ROE_Defination = ticker_name = ticker_current = stock_roe_ratio = roe_stock_benchmark = roe_stock_trend = \
roe_stock_comment = roe_summary = sma_defination = stock_sma_50 = sma_stock_benchmark = sma_stock_trend = \
sma_stock_comment = sma_summary = revenue_change_defination = stock_revenue = revenue_stock_benchmark = \
revenue_stock_trend = revenue_stock_comment = revenue_summary = debt_ratio_defination = stock_debt = \
stock_benchmark_debt = stock_trend_debt = stock_comment_debt = summary_debt = ticker_selected = None




def calculate_indicators(stock_symbol):
    #FIRST PB MODEL VARIABLES
    PB_Ratio_Defination = ticker_name = ticker_current = stock_pb_ratio = stock_benchmark = \
    stock_trend = stock_comment = pb_summary = ticker_selected = None
    
    #2nd MODEL ROE VARIABLES
    ROE_Defination = ticker_name = ticker_current = stock_roe_ratio = roe_stock_benchmark = \
    roe_stock_trend = roe_stock_comment = roe_summary = None
    
    #3rd Model SMA variables
    sma_defination = ticker_name= ticker_current= stock_sma_50 = sma_stock_benchmark= \
    sma_stock_trend= sma_stock_comment =sma_summary = None
    
    #4thmodel revenue model variables
    revenue_change_defination = ticker_name = ticker_current = stock_revenue = revenue_stock_benchmark = revenue_stock_trend = revenue_stock_comment = revenue_summary = None

    #5thmodel debth model
    debt_ratio_defination = ticker_name = ticker_current = stock_debt = stock_benchmark_debt = stock_trend_debt = stock_comment_debt = summary_debt = None  

    
    if stock_symbol:
        ticker_selected = stock_symbol
        print(ticker_selected, "SELECTED STOCK GOT THIS")
    
        #MODEL1 PB RATIO price debth ratio
        PB_Ratio_Defination, ticker_name, ticker_current, stock_pb_ratio ,stock_benchmark,\
            stock_trend, stock_comment, pb_summary  =   pb_ratio_model(ticker_selected)
        
        print("PB MODEL DONE")
        print(PB_Ratio_Defination, ticker_name, ticker_current, stock_pb_ratio ,stock_benchmark,\
            stock_trend, stock_comment, pb_summary)
        print("*****PB RATIO UP *****")

        stock_pb_ratio = str(stock_pb_ratio.split(" ")[0])
        
        
        #MODEL2 ROE Return of Equity
        ROE_Defination, ticker_name, ticker_current, stock_roe_ratio ,roe_stock_benchmark,\
            roe_stock_trend, roe_stock_comment, roe_summary = roe_model(ticker_selected)
        print(ROE_Defination, ticker_name, ticker_current, stock_roe_ratio ,roe_stock_benchmark,\
            roe_stock_trend, roe_stock_comment, roe_summary)
        print("***** ROE UP *****")
        
        
        
        #MODEL3 SMA Simple Moving Average
        sma_defination, ticker_name, ticker_current, stock_sma_50 ,sma_stock_benchmark,\
          sma_stock_trend, sma_stock_comment, sma_summary = sma_model(ticker_selected)
        print(sma_defination, ticker_name, ticker_current, stock_sma_50 ,sma_stock_benchmark,\
          sma_stock_trend, sma_stock_comment, sma_summary )
        print("***** SMA UP *****")
        
        
        #MODEL 4 REVENUE MODEL
        revenue_change_defination, ticker_name, ticker_current, stock_revenue ,revenue_stock_benchmark,\
          revenue_stock_trend, revenue_stock_comment, revenue_summary = revenue_model(ticker_selected)
        print(revenue_change_defination, ticker_name, ticker_current, stock_revenue ,revenue_stock_benchmark,\
          revenue_stock_trend, revenue_stock_comment, revenue_summary)
        print("**** REVENUE MODEL *****UP ")
        
        
        #MODEL 5 DEBTH  MODEL
        debt_ratio_defination, ticker_name, ticker_current, stock_debt ,stock_benchmark_debt,\
        stock_trend_debt, stock_comment_debt, summary_debt = debth_model(ticker_selected)
        print(  debt_ratio_defination, ticker_name, ticker_current, stock_debt ,stock_benchmark_debt,\
        stock_trend_debt, stock_comment_debt, summary_debt)
        print("**** DEBTH Up ******")
        
        
        
        

    return  {
    "PB_Ratio_Defination": PB_Ratio_Defination,
    "stock_pb_ratio": stock_pb_ratio, 
    "stock_benchmark": stock_benchmark,
    "stock_trend": stock_trend,
    "stock_comment": stock_comment,
    "pb_summary": pb_summary,
    
    # INDICATOR 2
    "ROE_Defination": ROE_Defination, 
    "ticker_name": ticker_name, 
    "ticker_current": ticker_current, 
    "stock_roe_ratio": stock_roe_ratio, 
    "roe_stock_benchmark": roe_stock_benchmark, 
    "roe_stock_trend": roe_stock_trend, 
    "roe_stock_comment": roe_stock_comment, 
    "roe_summary": roe_summary,
    
    # INDICATOR 3
    "sma_defination": sma_defination,
    "stock_sma_50": stock_sma_50,
    "sma_stock_benchmark": sma_stock_benchmark,
    "sma_stock_trend": sma_stock_trend, 
    "sma_stock_comment": sma_stock_comment, 
    "sma_summary": sma_summary,
    
    # INDICATOR 4
    "revenue_change_defination": revenue_change_defination,   
    "stock_revenue": stock_revenue,  
    "revenue_stock_benchmark": revenue_stock_benchmark,  
    "revenue_stock_trend": revenue_stock_trend,  
    "revenue_stock_comment": revenue_stock_comment,  
    "revenue_summary": revenue_summary,  
    
    # INDICATOR 5
    "debt_ratio_defination": debt_ratio_defination,  
    "stock_debt": stock_debt,  
    "stock_benchmark_debt": stock_benchmark_debt,  
    "stock_trend_debt": stock_trend_debt,  
    "stock_comment_debt": stock_comment_debt,  
    "summary_debt": summary_debt,  

    "ticker_selected": ticker_selected,
}

        



