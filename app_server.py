# api_server.py
# Example REST API server code
# This code is an example implementation of a REST-API endpoint that performs inference based on the TranAD_Gyeongsan model checkpoint.

from flask import Flask, request, jsonify
from datetime import datetime
from collections import deque, OrderedDict
from datetime import timedelta

import time
import torch
import numpy as np
import os
import sys
import logging
import src.dataprocessor as dp
import json
import shutil
import torch.nn as nn

# Import model classes
import src.models   # Add necessary models
import src.utils    # Assuming sensor data preprocessing functions are available

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Create an instance of DataProcessor (add initialization parameters if needed)
data_processor = dp.DataProcessor()

# -----------------------------------------------------------
# Model Loading Function Modification
# -----------------------------------------------------------
def load_model(model_name, model_path, input_dim):
    """
    Function to load the actual model.

    Args:
        model_name (str): Name of the model to load.
        model_path (str): Path to the model checkpoint file.
        input_dim (int): Dimension of the input data.

    Returns:
        torch.nn.Module: Loaded model.
    """    
    try:
        model_class = getattr(src.models, model_name)
    except AttributeError:
        logger.error(f"Model class '{model_name}' could not be found in src.models.")
        raise

    model = model_class(input_dim).double()
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"{model_name} model loaded successfully.")
    except Exception as e:
        logger.error(f"Error occurred while loading model checkpoint: {e}")
        raise
    return model

# -----------------------------------------------------------
# ThresholdManager Class Addition
# -----------------------------------------------------------
class ThresholdManager:
    """
    Class that accumulates anomaly_scores over a recent period (default: 1 year) to calculate thresholds.
    Users can choose a recent 3 months or 1 year based on options.
    """
    def __init__(self, window_months=12, storage_path="anomaly_scores.json"):
        """
        Initialization function.
        
        Args:
            window_months (int): Number of months to retain history (default: 12 months).
            storage_path (str): File path to store anomaly_scores.
        """
        self.window_months = window_months
        self.storage_path = storage_path
        self.anomaly_history = deque()
        self.load_history()

    def load_history(self):
        """
        Loads the stored anomaly_scores history.
        """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry in data:
                        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                        scores = entry["anomaly_scores"]
                        self.anomaly_history.append({"timestamp": timestamp, "scores": scores})
                self.prune_history()
                logger.info(f"Loaded {len(self.anomaly_history)} anomaly scores from history.")
            except Exception as e:
                logger.error(f"Error occurred while loading history: {e}")
                self.anomaly_history = deque()
        else:
            self.anomaly_history = deque()
            logger.info("History file does not exist. A new one will be created.")

    def save_history(self):
        """
        Saves the current anomaly_scores history.
        """
        try:
            # Save the current state even if anomaly_history is empty.
            data = [
                {
                    "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "anomaly_scores": entry["scores"]
                }
                for entry in self.anomaly_history
            ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved {len(data)} anomaly scores to history.")
        except Exception as e:
            logger.error(f"Error occurred while saving history: {e}")

    def prune_history(self):
        """
        Prunes the history to retain only data within the set period.
        """
        if self.anomaly_history:
            latest_timestamp = max(entry["timestamp"] for entry in self.anomaly_history)
            cutoff_date = latest_timestamp - timedelta(days=self.window_months * 30)  # Approximate month calculation
        else:
            cutoff_date = datetime.now() - timedelta(days=self.window_months * 30)  # Approximate month calculation
        initial_length = len(self.anomaly_history)
        while self.anomaly_history and self.anomaly_history[0]["timestamp"] < cutoff_date:
            self.anomaly_history.popleft()
        pruned_length = len(self.anomaly_history)
        logger.info(f"Pruned history: reduced from {initial_length} to {pruned_length} entries.")

    def add_anomaly_scores(self, timestamp, scores):
        """
        Adds new anomaly_scores to the history.
        
        Args:
            timestamp (datetime): Timestamp of the data.
            scores (list): List of anomaly_scores.
        """
        self.anomaly_history.append({"timestamp": timestamp, "scores": scores})
        logger.info(f"Added anomaly scores: {timestamp} - {scores}")
        self.prune_history()
        self.save_history()

    def calculate_thresholds(self, current_scores):
        """
        Calculates thresholds based on anomaly_scores over the set period.
        If no data exists, uses the current scores.
    
        Args:
            current_scores (numpy.ndarray): Current anomaly_scores
    
        Returns:
            numpy.ndarray: Calculated thresholds
        """
        if not self.anomaly_history:
            logger.warning("No data in history. Calculating thresholds based on current anomaly_scores.")
            mean_scores = np.mean(current_scores, axis=0)
            std_scores = np.std(current_scores, axis=0)
        else:
            all_scores = np.array([entry["scores"] for entry in self.anomaly_history])
            mean_scores = np.mean(all_scores, axis=0)
            std_scores = np.std(all_scores, axis=0)
        thresholds = mean_scores + 3 * std_scores
        logger.info(f"Calculated thresholds: {thresholds}")
        return thresholds

# -----------------------------------------------------------
# SensorManager Class Modification
# -----------------------------------------------------------
class SensorManager:
    """
    Class that manages models per sensor type and selects the appropriate model based on sensor ID.
    Utilizes an LRU cache to optimize memory usage.
    Automatically creates a sensor if an unknown sensor ID is requested.
    """
    def __init__(self, sensor_config_path="sensor_config.json", cache_size=100):
        """
        Initialization function.
        
        Args:
            sensor_config_path (str): Path to the JSON file storing sensor configurations.
            cache_size (int): Maximum number of models to keep in the cache.
        """
        self.sensor_config_path = sensor_config_path
        self.sensor_info = {}
        self.model_cache = OrderedDict()
        self.cache_size = cache_size
        self.load_sensor_config()

    def load_sensor_config(self):
        """
        Loads sensor configurations.
        """
        if not os.path.exists(self.sensor_config_path):
            logger.error(f"Sensor configuration file does not exist: {self.sensor_config_path}")
            sys.exit(1)
        
        try:
            with open(self.sensor_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            for sensor_id, sensor_details in config.items():
                self.sensor_info[sensor_id] = {
                    "type": sensor_details["type"],
                    "model_name": sensor_details["model_name"],
                    "model_version": sensor_details["model_version"],
                    "model_path": sensor_details["model_path"],
                    "input_dim": sensor_details["input_dim"]
                }
            logger.info("Sensor configurations loaded successfully.")
        except Exception as e:
            logger.error(f"Error occurred while loading sensor configurations: {e}")
            sys.exit(1)

    def save_sensor_config(self):
        """
        Saves sensor configurations to the JSON file.
        """
        try:
            config = {}
            for sensor_id, info in self.sensor_info.items():
                config[sensor_id] = {
                    "type": info["type"],
                    "model_name": info["model_name"],
                    "model_version": info["model_version"],
                    "model_path": info["model_path"],
                    "input_dim": info["input_dim"]
                }
            with open(self.sensor_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger.info("Sensor configuration file updated successfully.")
        except Exception as e:
            logger.error(f"Error occurred while saving sensor configuration: {e}")

    def add_new_sensor(self, sensor_id, sensor_type):
        """
        Adds a new sensor and updates the sensor_config.json file.
        
        Args:
            sensor_id (str): Unique ID of the new sensor.
            sensor_type (str): Type of the new sensor.
        """
        if sensor_type == "Tilt Sensor":
            model_name = "TranAD"
            model_version = "base_model_l"
            base_model_path = "/workspace/checkpoints/TranAD_Gyeongsan/base_model_l.ckpt"
            new_model_filename = f"TiltSensor_{sensor_id}.ckpt"
            new_model_path = f"/workspace/checkpoints/TranAD_Gyeongsan/{new_model_filename}"
            input_dim = 9
        elif sensor_type == "Crack Sensor":
            model_name = "TranAD"
            model_version = "base_model_c"
            base_model_path = "/workspace/checkpoints/TranAD_Gyeongsan/base_model_c.ckpt"
            new_model_filename = f"CrackSensor_{sensor_id}.ckpt"
            new_model_path = f"/workspace/checkpoints/TranAD_Gyeongsan/{new_model_filename}"
            input_dim = 7
        else:
            logger.error(f"Unsupported sensor type: {sensor_type}")
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
        
        # Copy the base model to a new model file
        try:
            shutil.copyfile(base_model_path, new_model_path)
            logger.info(f"Copied base model from {base_model_path} to {new_model_path}")
        except Exception as e:
            logger.error(f"Failed to copy base model: {e}")
            raise
        
        self.sensor_info[sensor_id] = {
            "type": sensor_type,
            "model_name": model_name,
            "model_version": f"{sensor_type}_{sensor_id}",
            "model_path": new_model_path,
            "input_dim": input_dim
        }
        self.save_sensor_config()
        logger.info(f"Added new sensor: {sensor_id} ({sensor_type}) with model {new_model_path}")

    def get_model(self, sensor_id, sensor_type=None):
        """
        Returns the model corresponding to the sensor ID. Loads dynamically if necessary.
        Automatically adds the sensor if the sensor ID is unknown.
        
        Args:
            sensor_id (str): Unique ID of the sensor.
            sensor_type (str, optional): Type of the sensor. Required when adding a new sensor.
        
        Returns:
            dict: Sensor information including the loaded model.
        """
        if sensor_id not in self.sensor_info:
            if sensor_type is None:
                logger.error(f"Sensor type not provided for unknown sensor ID: {sensor_id}")
                raise ValueError("Unknown sensor ID and sensor type not provided.")
            logger.info(f"Unknown sensor ID requested: {sensor_id}. Adding new sensor.")
            self.add_new_sensor(sensor_id, sensor_type)
        
        sensor = self.sensor_info[sensor_id]
        model_key = f"{sensor['type']}_{sensor['model_version']}"
        
        # Check if the model is already in the cache
        if model_key in self.model_cache:
            model = self.model_cache.pop(model_key)
            # Move the recently used model to the end of the cache
            self.model_cache[model_key] = model
            return {"sensor_info": sensor, "model": model}
        else:
            # Load the model and add it to the cache
            model = load_model(sensor['model_name'], sensor['model_path'], sensor['input_dim'])
            self.model_cache[model_key] = model
            logger.info(f"Loaded and cached model: {model_key}")
            # Remove the least recently used model if cache exceeds capacity
            if len(self.model_cache) > self.cache_size:
                removed_key, removed_model = self.model_cache.popitem(last=False)
                del removed_model
                logger.info(f"Cache capacity exceeded. Removed model: {removed_key}")
            return {"sensor_info": sensor, "model": model}

# -----------------------------------------------------------
# SensorManager Instance Initialization Modification
# -----------------------------------------------------------
sensor_manager = SensorManager(
    sensor_config_path=os.path.join(os.path.dirname(__file__), "sensor_config.json"),
    cache_size=100  # Set cache capacity as needed
)

# -----------------------------------------------------------
# ThresholdManager 인스턴스 초기화 추가
# -----------------------------------------------------------
threshold_manager = ThresholdManager()

# -----------------------------------------------------------
# Inference Endpoint Modification
# -----------------------------------------------------------
@app.route("/v1/inference/sensor-check", methods=["POST"])
def inference_endpoint():
    """
    API endpoint that performs inference based on sensor data.
    
    Request Format:
    {
        "deviceId": "Sensor ID",
        "sensor_type": "Sensor Type",
        "payload": { ... },
        "@timestamp": "YYYY-MM-DD HH:MM:SS"
    }
    
    Response Format:
    {
        "data": {
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "inference_result": {
                "status": "True" or "False",
                "anomaly_scores": [float, ...],
                "thresholds": [float, ...],
                "prediction": [float, ...]
            },
            "model_info": {
                "version": "Model Version",
                "inference_time_ms": float
            }
        }
    }
    """
    start_time = time.time()
    req_data = request.get_json()
    
    if req_data is None:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": "Please provide valid JSON data."}}), 400

    if "deviceId" not in req_data:
        return jsonify({"error": {"code": "MISSING_FIELD", "message": "Missing 'deviceId' field."}}), 400

    if "sensor_type" not in req_data:
        return jsonify({"error": {"code": "MISSING_FIELD", "message": "Missing 'sensor_type' field."}}), 400

    device_id = req_data["deviceId"]
    sensor_type = req_data["sensor_type"]

    try:
        sensor_info = sensor_manager.get_model(device_id, sensor_type)
        sensor_type = sensor_info["sensor_info"]["type"]
    except ValueError as ve:
        return jsonify({"error": {"code": "INVALID_SENSOR_TYPE", "message": str(ve)}}), 400
    except Exception as e:
        return jsonify({"error": {"code": "SERVER_ERROR", "message": "An internal server error occurred."}}), 500

    # Validate request data
    valid, err_msg = validate_input(req_data, sensor_type)
    if not valid:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": err_msg}}), 400
    
    payload = req_data["payload"]
    timestamp_value = req_data["@timestamp"]
    
    # Extract sensor values based on sensor type
    if sensor_type == "Tilt Sensor":
        sensor_values = {
            "initDegreeX": payload.get("initDegreeX"),
            "initDegreeY": payload.get("initDegreeY"),
            "degreeXAmount": payload.get("degreeXAmount"),
            "degreeYAmount": payload.get("degreeYAmount"),
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity")
        }
    elif sensor_type == "Crack Sensor":
        sensor_values = {
            "initCrack": payload.get("initCrack"),
            "crackAmount": payload.get("crackAmount"),
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity")
        }
    else:
        sensor_values = {}

    # Perform model inference
    status, prediction, anomaly_scores, thresholds = model_inference(device_id, sensor_values, timestamp_value)

    # Measure inference time (ms)
    inference_time_ms = (time.time() - start_time) * 1000.0

    # Construct response JSON
    response = {
        "data": {
            "timestamp": timestamp_value,
            "inference_result": {
                "status": str(status),
                "anomaly_scores": anomaly_scores.tolist() if isinstance(anomaly_scores, np.ndarray) else anomaly_scores,
                "thresholds": thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds,
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
            },
            "model_info": {
                "version": f"{sensor_info['sensor_info']['type']} {sensor_info['sensor_info']['model_version']}",
                "inference_time_ms": round(inference_time_ms, 2)
            }
        }
    }
    logger.info(f"Response generated: {response}")
    return jsonify(response), 200

# -----------------------------------------------------------
# Validate Input Function Modification
# -----------------------------------------------------------
def validate_input(req_data, sensor_type):
    """
    Validates the incoming request data based on the sensor type.
    
    Args:
        req_data (dict): The incoming request data.
        sensor_type (str): The type of the sensor.
        
    Returns:
        tuple: (is_valid (bool), error_message (str))
    """
    required_fields = []
    if sensor_type == "Tilt Sensor":
        required_fields = ["initDegreeX", "initDegreeY", "degreeXAmount", "degreeYAmount", "temperature", "humidity"]
    elif sensor_type == "Crack Sensor":
        required_fields = ["initCrack", "crackAmount", "temperature", "humidity"]
    else:
        return False, "Unsupported sensor type."
    
    missing_fields = [field for field in required_fields if field not in req_data["payload"]]
    if missing_fields:
        return False, f"Missing fields in payload: {', '.join(missing_fields)}"
    
    return True, ""

# -----------------------------------------------------------
# API Endpoint to Read and Modify window_months Configuration
# -----------------------------------------------------------
@app.route("/v1/config/window_months", methods=["GET", "POST"])
def manage_window_months():
    """
    API endpoint to read or modify the window_months setting.
    
    GET Request:
        - Returns the current window_months value.
    
    POST Request:
        - Includes the "window_months" field in the JSON body to modify the value.
        - Example JSON body:
          {
              "window_months": 6
          }
    
    Response Format:
        {
            "window_months": Current Set Value
        }
    """
    if request.method == "GET":
        return jsonify({"window_months": threshold_manager.window_months}), 200
    
    elif request.method == "POST":
        req_data = request.get_json()
        if req_data is None:
            return jsonify({"error": {"code": "INVALID_INPUT", "message": "Please provide valid JSON data."}}), 400
        
        if "window_months" not in req_data:
            return jsonify({"error": {"code": "MISSING_FIELD", "message": "'window_months' field is required."}}), 400
        
        new_window = req_data["window_months"]
        
        if not isinstance(new_window, int) or new_window <= 0:
            return jsonify({"error": {"code": "INVALID_VALUE", "message": "'window_months' must be a positive integer."}}), 400
        
        try:
            threshold_manager.window_months = new_window
            threshold_manager.prune_history()  # Prune history based on the new period
            threshold_manager.save_history()   # Save the updated settings
            logger.info(f"window_months has been updated to {new_window}.")
            return jsonify({"window_months": new_window}), 200
        except Exception as e:
            logger.error(f"Error occurred while updating window_months: {e}")
            return jsonify({"error": {"code": "SERVER_ERROR", "message": "An error occurred while updating the settings."}}), 500

# # -----------------------------------------------------------
# # model_inference 함수 수정
# # -----------------------------------------------------------
# def model_inference(device_id, sensor_values, timestamp):
#     """
#     센서에 적합한 모델을 사용하여 추론을 수행하는 함수.
    
#     Args:
#         device_id (str): 디바이스 ID.
#         sensor_values (dict): 센서 데이터.
#         timestamp (str): 데이터의 타임스탬프.
    
#     Returns:
#         tuple: (status, prediction, anomaly_scores, thresholds)
#     """
#     try:
#         # timestamp 파싱 및 시간 관련 특성 추출
#         dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
#         hour = dt.hour
#         day_of_week = dt.weekday()
#         month = dt.month

#         # 시간 관련 특성 추가
#         sensor_values_extended = {**sensor_values, "hour": hour, "day_of_week": day_of_week, "month": month}

#         # 확장된 센서 데이터 처리
#         sensor_data = np.array(list(sensor_values_extended.values()))
#         model_info = sensor_manager.get_model(device_id, sensor_values.get("type"))
#         model = model_info["model"]
#         sensor_type = model_info["sensor_info"]["type"]
        
#         input_tensor = torch.tensor(sensor_data, dtype=torch.double)
#         prediction = model(input_tensor)
#         anomaly_scores = prediction.detach().numpy()
        
#         thresholds = threshold_manager.calculate_thresholds(anomaly_scores)
#         status = (anomaly_scores > thresholds).any()
        
#         # anomaly_scores를 threshold_manager에 추가
#         threshold_manager.add_anomaly_scores(dt, anomaly_scores.tolist())
        
#         return status, prediction, anomaly_scores, thresholds
#     except Exception as e:
#         logger.error(f"모델 추론 중 오류 발생: {e}")
#         raise

# -----------------------------------------------------------
# preprocess_sensor_data 함수 추가
# -----------------------------------------------------------
def preprocess_sensor_data(sensor_values_extended, input_dim):
    """
    센서 데이터를 전처리하는 함수.

    Args:
        sensor_values_extended (dict): 확장된 센서 데이터.
        input_dim (int): 센서 데이터의 차원 (9 또는 7).

    Returns:
        tuple: 전처리된 window 텐서과 elem 텐서.
    """
    try:
        # 센서 데이터를 NumPy 배열로 변환
        sensor_array = np.array(list(sensor_values_extended.values()), dtype=np.float64)

        # 데이터 정규화
        try:
            normalized_data, _, _ = data_processor.normalize3(sensor_array)
        except Exception as e:
            logger.error(f"데이터 정규화 중 오류 발생: {e}")
            raise

        # 정규화된 데이터를 텐서로 변환
        sensor_tensor = torch.from_numpy(normalized_data).double().unsqueeze(0)  # Shape: [1, 13]
        sensor_tensor = sensor_tensor.expand(10, -1)  # Shape: [10, 13]

        window = sensor_tensor.unsqueeze(1)  # Shape: [10, 1, 13]
        elem = window[-1, :, :].view(1, 1, input_dim)  # Shape: [1, 1, 13]

        return window, elem
    except Exception as e:
        logger.error(f"preprocess_sensor_data 중 오류 발생: {e}")
        raise

# -----------------------------------------------------------
# model_inference 함수 수정
# -----------------------------------------------------------
def model_inference(device_id, sensor_values, timestamp):
    """
    센서에 적합한 모델을 사용하여 추론을 수행하는 함수.
    
    Args:
        device_id (str): 디바이스 ID.
        sensor_values (dict): 센서 데이터.
        timestamp (str): 데이터의 타임스탬프.
    
    Returns:
        tuple: (status, prediction, anomaly_scores, thresholds)
    """
    try:
        # timestamp 파싱 및 시간 관련 특성 추출
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month

        # 시간 관련 특성 추가
        sensor_values_extended = {
            **sensor_values,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month
        }

        # 모델에 window와 elem 입력
        model_info = sensor_manager.get_model(device_id, sensor_values.get("type"))
        model = model_info["model"]
        sensor_type = model_info["sensor_info"]["type"]
        input_dim = model_info['sensor_info']['input_dim']

        # 확장된 센서 데이터 전처리
        window, elem = preprocess_sensor_data(sensor_values_extended, input_dim)  # 입력 차원에 맞게 조정 (예: 9 또는 7)
        output = model(window, elem)
        if isinstance(output, tuple):
            output = output[1]

        # MSE Loss 계산
        mse_loss = nn.MSELoss(reduction='none')
        anomaly_score = mse_loss(output, elem)[0].detach().cpu().numpy()

        # Threshold 계산
        thresholds = threshold_manager.calculate_thresholds(anomaly_score)

        # 상태 결정
        status = (anomaly_score > thresholds).any()

        # anomaly_score를 threshold_manager에 추가
        threshold_manager.add_anomaly_scores(dt, anomaly_score.tolist())

        # 모델 예측 결과
        prediction = output.detach().cpu().numpy().flatten()

        return status, prediction, anomaly_score, thresholds
    except Exception as e:
        logger.error(f"모델 추론 중 오류 발생: {e}")
        raise
    

# -----------------------------------------------------------
# Main Execution Block Modification
# -----------------------------------------------------------
def run_server():
    """
    Function to run the Flask server.
    """
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Set werkzeug's log level to ERROR to suppress WARNING messages

    try:
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 8080))
        debug = os.getenv('FLASK_DEBUG', 'False') == 'True'  # Set debug mode to True by default

        app.run(host=host, port=port, debug=debug, use_reloader=True)  # Enable auto reloader
    except Exception as e:
        logger.error(f"Error occurred while running the server: {e}")

if __name__ == "__main__":
    run_server()
