from flask import Flask, send_from_directory, abort, request
from stock_home_1 import get_predictions
import os
import json
import datetime
import dotenv
import logging
import requests
from dateutil.parser import parse

dotenv.load_dotenv()

STOCK_MODELS_DIR = os.getenv("STOCK_MODELS_DIR")
STOCKS_SUMMARY_PATH = os.getenv("STOCKS_SUMMARY_PATH")

STOCKS_PLOT_PATH = os.getenv("STOCKS_PLOT_PATH")
IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')
IMGUR_HOST_URL = 
STOCKS_DATA_PATH = os.getenv("STOCKS_DATA_PATH")
time_format = "%Y-%m-%dT%H:%M:%S"

app = Flask(__name__)

# Logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

## View logging in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def hostImageImgur(image_path):
    headers = {
        'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'
    }

    files = {
        'image': open(image_path, 'rb')
    }

    data = {
        'title': 'fileA', 
        'type': 'file',
        'description': 'A simple upload',  

    }

    response = requests.post(IMGUR_HOST_URL, headers=headers, files=files, data=data)
    
    return response.json().get('data', {}).get('link', 'No link found')

def log_api_call(request):
    timestamp = datetime.datetime.now()
    logging.info(f"API CALL: /download_image | Method: {request.method} | IP: {request.remote_addr} | Stock: {request.args.get('stockname')} | Time: {timestamp}")

def generate_data(stockname, stockname_model_name, write_to_file:bool=True) -> dict:
    current_time = datetime.datetime.now().strftime(time_format)
    logging.info(f"Updating data for stockname: {stockname}...")

    # Get predictions
    _, _, _, image_path, summary, _ = get_predictions(stockname_model_name)

    # Host image on Imgur
    image_imgur_url = hostImageImgur(image_path)

    stock_data = {
            "model_name": stockname_model_name,
            "summary_content": summary,
            "image_imgur_url": image_imgur_url,
            "image_local_name":  os.path.basename(image_path),
            "latest_update": current_time
        }
    if write_to_file:
        # Read existing data first
        with open(STOCKS_DATA_PATH, 'r') as f:
            all_stock_data = json.load(f)
            all_stock_data[stockname] = stock_data

        # Write to file
        os.makedirs(os.path.dirname(STOCKS_DATA_PATH), exist_ok=True)
        with open(STOCKS_DATA_PATH, 'w') as f:
            json.dump(all_stock_data, f)
        logging.info(f"Data for {stockname} written to file successfully.")

    return stock_data

def rebuild_stock_data_file():
    stock_data = {
        "btc": {
            "model_name": "BTC-USD",
            "summary_content": "",
            "image_imgur_url": "",
            "image_local_name": "",
            "latest_update": ""
        },
        "eth": {
            "model_name": "ETH-USD",
            "summary_content": "",
            "image_imgur_url": "",
            "image_local_name": "",
            "latest_update": ""
        }
    }

    os.makedirs(os.path.dirname(STOCKS_DATA_PATH), exist_ok=True)
    with open(STOCKS_DATA_PATH, 'w') as f:
        json.dump(stock_data, f)

    return stock_data

@app.route('/tst', methods=['GET'])
def tst():
    log_api_call(request)

    # Stockname data file validation
    if os.path.exists(STOCKS_DATA_PATH):
        try:
            with open(STOCKS_DATA_PATH, 'r') as f:
                stock_data = json.load(f)
        except Exception as e:
            logging.error(f"[INTERNAL] Error reading stock data file: {e}.. rebuilding stock data file")
            stock_data = rebuild_stock_data_file()
    else :
        logging.info(f"Stock data file not found at {STOCKS_DATA_PATH}, creating new data")
        stock_data = rebuild_stock_data_file()

    # Stockname validation
    request_stockname = request.args.get("stockname").lower()

    if not request_stockname or request_stockname not in stock_data.keys():
        logging.error("[CLIENT] stockname is required and must be one of: btc, eth")
        return json.dumps({"error-client": "stockname is required and must be one of: btc, eth"}), 400
    
    else:
        # Get stockname model name
        stockname = request_stockname
        stockname_model_name = stock_data[request_stockname]["model_name"]
        stockname_data = stock_data[request_stockname]

        # Check last update time
        current_time = datetime.datetime.now()
        if "latest_update" in stockname_data and stockname_data["latest_update"] != "":
            last_update_time = datetime.datetime.strptime(stockname_data["latest_update"],time_format)
        else:
            last_update_time = datetime.datetime.min
        time_diff = (current_time - last_update_time).total_seconds() / 60
        
        # Check if last update time is more than 60 minutes ago or can't be found
        if not last_update_time or time_diff > 60:  
            logging.info("Last update was more than 60 minutes ago, proceeding with update")
            new_stock_data = generate_data(stockname, stockname_model_name, write_to_file=True)
            
            logging.info(f"New stock data generated for {stockname}")
            return json.dumps(new_stock_data), 200
        else :
            logging.info("Last update was within 60 minutes, no need to update data")
            return json.dumps(stockname_data), 200

       

@app.route('/download_image', methods=['GET'])
def download_image():
    # get stockname and info
    stockname = request.args.get("stockname")

    summary_file_path = f"{STOCKS_SUMMARY_PATH}/{str(stockname)}_summary.txt"

    if os.path.isfile(summary_file_path):

        # get last update time
        timestamp = os.path.getmtime(STOCKS_SUMMARY_PATH)
        last_modified_time = datetime.datetime.fromtimestamp(timestamp)

        # get current time & diff
        current_time = datetime.datetime.now()
        time_diff = (current_time - last_modified_time).total_seconds() / 60

        if time_diff <= 50:
            logging.info("Last update was within 50 minutes, no need to update")
            img_file_path = f"{STOCKS_PLOT_PATH}/{str(stockname)}_prediction_plot.jpg"
            file_path, filename = os.path.split(img_file_path)
            return send_from_directory(file_path, filename, as_attachment=True)

        os.remove(summary_file_path)
        logging.info("Summary file found and removed to be updated")

    a, b, c, file_path, _, _ = get_predictions(stockname)
    logging.info(file_path)
    # download plot
    if not os.path.isfile(file_path):
        logging.error("File not found")
        return json.dumps({
            "error_message": "File not found, please wait a few minutes for the file to be generated.",
        }), 400

    file_path, filename = os.path.split(file_path)
    return send_from_directory(file_path, filename, as_attachment=True)


@app.route('/download_summary', methods=['GET'])
def download_summary():
    # get stockname and info
    stockname = request.args.get("stockname")
    user_ip = request.remote_addr
    method = request.method
    timestamp = datetime.datetime.now().isoformat()


    # monitor calls
    logging.info(
        f"API CALL: /download_summary | Method: {method} | IP: {user_ip} | Stock: {stockname} | Time: {timestamp}")

    summary_file_path = f"{STOCKS_SUMMARY_PATH}/{str(stockname)}_summary.txt"

    # get last update time
    timestamp = os.path.getmtime(STOCKS_SUMMARY_PATH)
    last_modified_time = datetime.datetime.fromtimestamp(timestamp)
    latest_update = last_modified_time.strftime("%Y_%m_%d_%H_%M")

    # read summary
    if not os.path.isfile(summary_file_path):
        logging.error("File not found")
        return json.dumps({
            "error_message": "File not found, please generate the image first, or wait a few minutes for the file to be generated.",
        }), 400
    
    
    with open(summary_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    output = {
        "summary": content,
        "latestUpdate": latest_update
    }

    # return in json format
    return json.dumps(output), 200


if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=5007, debug=True)