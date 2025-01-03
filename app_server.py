# api_server.py
# Example REST API server code
# This code is an example implementation of a REST-API endpoint that performs inference based on the TranAD_Gyeongsan model checkpoint.

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from collections import deque, OrderedDict, defaultdict

import time
import torch
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch.nn as nn

# Import custom modules
import src.dataprocessor as dp
import src.models
import src.utils  # Assuming sensor data preprocessing functions are available

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# 파일 핸들러 추가
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 데이터 프로세서 인스턴스 생성
data_processor = dp.DataProcessor()

# 최대 및 최소값 파일 경로
MAX_CRACK_PATH = "/workspace/processed/Gyeongsan_c/max_crack.npy"
MIN_CRACK_PATH = "/workspace/processed/Gyeongsan_c/min_crack.npy"
MAX_SLOPE_PATH = "/workspace/processed/Gyeongsan_s/max_slope.npy"
MIN_SLOPE_PATH = "/workspace/processed/Gyeongsan_s/min_slope.npy"

# 최대 및 최소값 로드
try:
    max_crack = np.load(MAX_CRACK_PATH)
    min_crack = np.load(MIN_CRACK_PATH)
    max_slope = np.load(MAX_SLOPE_PATH)
    min_slope = np.load(MIN_SLOPE_PATH)
    logger.info("최대 및 최소값 파일이 성공적으로 로드되었습니다.")
except Exception as e:
    logger.error(f"최대/최소값 파일 로드 중 오류 발생: {e}")
    sys.exit(1)

# -----------------------------------------------------------
# 정규화 해제 함수 정의
# -----------------------------------------------------------
def denormalize(values, max_vals, min_vals):
    """
    정규화된 값을 실제 값으로 변환합니다.
    
    Args:
        values (list or np.ndarray): 정규화된 값 리스트.
        max_vals (np.ndarray): 각 특성의 최대값 배열.
        min_vals (np.ndarray): 각 특성의 최소값 배열.
    
    Returns:
        list: 실제 값 리스트.
    """
    values = np.array(values)
    actual_values = values * (max_vals - min_vals) + min_vals
    return actual_values.tolist()

# -----------------------------------------------------------
# 모델 로딩 함수
# -----------------------------------------------------------
def load_model(model_name, model_path, input_dim):
    """
    실제 모델을 로드하는 함수.

    Args:
        model_name (str): 로드할 모델의 이름.
        model_path (str): 모델 체크포인트 파일의 경로.
        input_dim (int): 입력 데이터의 차원.

    Returns:
        torch.nn.Module: 로드된 모델.
    """
    try:
        model_class = getattr(src.models, model_name)
    except AttributeError:
        logger.error(f"src.models에 '{model_name}' 클래스가 존재하지 않습니다.")
        raise

    model = model_class(input_dim).double()
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"{model_name} 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        logger.error(f"모델 체크포인트 로딩 중 오류 발생: {e}")
        raise
    return model

# -----------------------------------------------------------
# ThresholdManager 클래스 수정
# -----------------------------------------------------------
class ThresholdManager:
    """
    각 모델별로 anomaly_scores를 누적하여 thresholds를 계산하는 클래스.
    """
    def __init__(self, window_months=3, storage_path="anomaly_scores.json"):
        """
        초기화 함수.

        Args:
            window_months (int): 유지할 기간의 개월 수 (기본: 12개월).
            storage_path (str): anomaly_scores를 저장할 파일 경로.
        """
        self.window_months = window_months
        self.storage_path = storage_path
        self.anomaly_history = defaultdict(deque)  # 모델별로 deque를 관리
        self.load_history()

    def load_history(self):
        """
        저장된 anomaly_scores 히스토리를 로드.
        """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for model_name, entries in data.items():
                        for entry in entries:
                            timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                            scores = entry["anomaly_scores"]
                            self.anomaly_history[model_name].append({"timestamp": timestamp, "scores": scores})
                self.prune_history()
                logger.info(f"히스토리에서 {len(self.anomaly_history)}개의 모델에 대한 anomaly scores를 로드했습니다.")
            except Exception as e:
                logger.error(f"히스토리 로드 중 오류 발생: {e}")
                self.anomaly_history = defaultdict(deque)
        else:
            self.anomaly_history = defaultdict(deque)
            logger.info("히스토리 파일이 존재하지 않습니다. 새로 생성됩니다.")

    def save_history(self):
        """
        현재 anomaly_scores 히스토리를 저장.
        """
        try:
            data = {
                model_name: [
                    {
                        "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "anomaly_scores": entry["scores"]
                    }
                    for entry in entries
                ]
                for model_name, entries in self.anomaly_history.items()
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"히스토리에 {len(data)}개의 모델 anomaly scores를 저장했습니다.")
        except Exception as e:
            logger.error(f"히스토리 저장 중 오류 발생: {e}")

    def prune_history(self):
        """
        설정된 기간 내의 데이터만 유지하도록 히스토리를 정리.
        """
        cutoff_date = datetime.now() - timedelta(days=self.window_months * 30)  # 개월을 일수로 근사 계산
        for model_name, entries in self.anomaly_history.items():
            initial_length = len(entries)
            while entries and entries[0]["timestamp"] < cutoff_date:
                entries.popleft()
            pruned_length = len(entries)
            logger.info(f"모델 '{model_name}'의 히스토리를 {initial_length}에서 {pruned_length}로 정리했습니다.")

    def add_anomaly_scores(self, model_basename, timestamp, scores):
        """
        특정 모델의 anomaly_scores를 히스토리에 추가.

        Args:
            model_basename (str): 모델의 기본 이름.
            timestamp (str): 타임스탬프 문자열.
            scores (list): anomaly_scores 리스트.
        """
        try:
            timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            self.anomaly_history[model_basename].append({"timestamp": timestamp_dt, "scores": scores})
            self.prune_history()
            self.save_history()
            logger.info(f"모델 '{model_basename}'의 anomaly_scores가 히스토리에 추가되었습니다.")
        except Exception as e:
            logger.error(f"anomaly_scores 추가 중 오류 발생: {e}")

    def calculate_thresholds(self, model_basename, current_scores):
        """
        특정 모델의 anomaly_scores를 기반으로 thresholds를 계산.
        데이터가 없으면 현재 scores로 계산.

        Args:
            model_basename (str): 모델의 기본 이름.
            current_scores (numpy.ndarray): 현재 anomaly_scores.

        Returns:
            numpy.ndarray: 계산된 thresholds.
        """
        if not self.anomaly_history[model_basename]:
            logger.warning(f"모델 '{model_basename}'의 히스토리에 데이터가 없습니다. 현재 anomaly_scores로 thresholds를 계산합니다.")
            mean_scores = np.mean(current_scores, axis=0)
            std_scores = np.std(current_scores, axis=0)
        else:
            all_scores = np.array([entry["scores"] for entry in self.anomaly_history[model_basename]])
            mean_scores = np.mean(all_scores, axis=0)
            std_scores = np.std(all_scores, axis=0)
        thresholds = mean_scores + 3 * std_scores
        logger.info(f"모델 '{model_basename}'의 thresholds를 계산했습니다: {thresholds}")
        return thresholds

# -----------------------------------------------------------
# SensorManager 클래스 수정
# -----------------------------------------------------------
class SensorManager:
    """
    센서 유형별로 모델을 관리하고 센서 ID에 따라 적절한 모델을 선택하는 클래스.
    LRU 캐시를 사용하여 메모리 사용을 최적화합니다.
    알려지지 않은 센서 ID 요청 시 자동으로 센서를 생성합니다.
    """
    def __init__(self, sensor_config_path="sensor_config.json", cache_size=100):
        """
        초기화 함수.

        Args:
            sensor_config_path (str): 센서 구성을 저장하는 JSON 파일의 경로.
            cache_size (int): 캐시에 저장할 모델의 최대 수.
        """
        self.sensor_config_path = sensor_config_path
        self.sensor_info = {}
        self.model_cache = OrderedDict()
        self.cache_size = cache_size
        self.load_sensor_config()

    def load_sensor_config(self):
        """
        센서 구성을 로드합니다. 파일이 없으면 새로 생성합니다.
        """
        if not os.path.exists(self.sensor_config_path):
            logger.info(f"센서 구성 파일이 존재하지 않습니다. 새로운 파일을 생성합니다: {self.sensor_config_path}")
            try:
                with open(self.sensor_config_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
                logger.info(f"새로운 센서 구성 파일이 생성되었습니다: {self.sensor_config_path}")
            except Exception as e:
                logger.error(f"센서 구성 파일 생성 중 오류 발생: {e}")
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
            logger.info("센서 구성이 성공적으로 로드되었습니다.")
        except Exception as e:
            logger.error(f"센서 구성 로딩 중 오류 발생: {e}")
            sys.exit(1)

    def save_sensor_config(self):
        """
        센서 구성을 JSON 파일에 저장합니다.
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
            logger.info("센서 구성 파일이 성공적으로 업데이트되었습니다.")
        except Exception as e:
            logger.error(f"센서 구성 저장 중 오류 발생: {e}")

    def add_new_sensor(self, sensor_id, sensor_type):
        """
        새로운 센서를 추가하고 센서 구성을 업데이트합니다.

        Args:
            sensor_id (str): 새로운 센서의 고유 ID.
            sensor_type (str): 새로운 센서의 유형.
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
            logger.error(f"지원하지 않는 센서 유형입니다: {sensor_type}")
            raise ValueError(f"지원하지 않는 센서 유형입니다: {sensor_type}")

        # 기본 모델을 새 모델 파일로 복사
        try:
            shutil.copyfile(base_model_path, new_model_path)
            logger.info(f"기본 모델을 {base_model_path}에서 {new_model_path}로 복사했습니다.")
        except Exception as e:
            logger.error(f"기본 모델 복사 실패: {e}")
            raise

        self.sensor_info[sensor_id] = {
            "type": sensor_type,
            "model_name": model_name,
            "model_version": model_version,
            "model_path": new_model_path,
            "input_dim": input_dim
        }
        self.save_sensor_config()
        logger.info(f"새 센서가 추가되었습니다: {sensor_id} ({sensor_type}) - 모델 경로: {new_model_path}")

    def get_model(self, sensor_id, sensor_type=None):
        """
        센서 ID에 해당하는 모델을 반환합니다. 필요 시 동적으로 로드합니다.
        알려지지 않은 센서 ID인 경우 자동으로 센서를 추가합니다.

        Args:
            sensor_id (str): 센서의 고유 ID.
            sensor_type (str, optional): 센서의 유형. 새 센서 추가 시 필요.

        Returns:
            dict: 센서 정보와 로드된 모델을 포함한 사전.
        """
        if sensor_id not in self.sensor_info:
            if sensor_type is None:
                logger.error(f"알려지지 않은 센서 ID입니다: {sensor_id}. 센서 유형이 제공되지 않았습니다.")
                raise ValueError("알려지지 않은 센서 ID이며, 센서 유형이 제공되지 않았습니다.")
            logger.info(f"알려지지 않은 센서 ID 요청됨: {sensor_id}. 새로운 센서를 추가합니다.")
            self.add_new_sensor(sensor_id, sensor_type)

        sensor = self.sensor_info[sensor_id]
        model_key = f"{sensor['type']}_{sensor['model_version']}"

        # 캐시에 모델이 있는지 확인
        if model_key in self.model_cache:
            model = self.model_cache.pop(model_key)
            self.model_cache[model_key] = model  # 최근 사용된 모델을 캐시의 끝으로 이동
            return {"sensor_info": sensor, "model": model}
        else:
            # 모델 로드 및 캐시에 추가
            model = load_model(sensor['model_name'], sensor['model_path'], sensor['input_dim'])
            self.model_cache[model_key] = model
            logger.info(f"모델을 로드하고 캐시에 추가했습니다: {model_key}")
            # 캐시 용량 초과 시 가장 오래된 모델 제거
            if len(self.model_cache) > self.cache_size:
                removed_key, removed_model = self.model_cache.popitem(last=False)
                del removed_model
                logger.info(f"캐시 용량 초과로 모델을 제거했습니다: {removed_key}")
            return {"sensor_info": sensor, "model": model}

# -----------------------------------------------------------
# SensorManager 인스턴스 초기화
# -----------------------------------------------------------
sensor_manager = SensorManager(
    sensor_config_path=os.path.join(os.path.dirname(__file__), "sensor_config.json"),
    cache_size=100  # 필요에 따라 캐시 용량 설정
)

# -----------------------------------------------------------
# ThresholdManager 인스턴스 초기화
# -----------------------------------------------------------
threshold_manager = ThresholdManager()

# -----------------------------------------------------------
# 센서 데이터 전처리 함수
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
        elem = window[-1, :, :].view(1, 1, input_dim)  # Shape: [1, 1, input_dim]

        return window, elem
    except Exception as e:
        logger.error(f"preprocess_sensor_data 중 오류 발생: {e}")
        raise

# -----------------------------------------------------------
# 모델 추론 함수
# -----------------------------------------------------------
def model_inference(device_id, sensor_values, timestamp):
    """
    모델을 사용하여 추론을 수행하고 결과를 반환합니다.

    Args:
        device_id (str): 디바이스 ID.
        sensor_values (dict): 센서 값.
        timestamp (str): 타임스탬프.

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
        model_version = model_info['sensor_info']['model_version']
        model_path = model_info["sensor_info"]["model_path"]
        model_basename = os.path.splitext(os.path.basename(model_path))[0]  # 모델 경로의 베이스네임 추출

        # 확장된 센서 데이터 전처리
        window, elem = preprocess_sensor_data(sensor_values_extended, input_dim)
        output = model(window, elem)
        if isinstance(output, tuple):
            output = output[1]

        # MSE Loss 계산
        mse_loss = nn.MSELoss(reduction='none')
        anomaly_score = mse_loss(output, elem)[0].detach().cpu().numpy()

        # Threshold 계산
        thresholds = threshold_manager.calculate_thresholds(model_basename, anomaly_score)

        # 상태 결정
        status = (anomaly_score[:,:-3] > thresholds[:,:-3]).any()

        # anomaly_score를 threshold_manager에 추가
        threshold_manager.add_anomaly_scores(model_basename, dt, anomaly_score.tolist())

        # 모델 예측 결과
        prediction = output.detach().cpu().numpy().flatten()

        return status, prediction, anomaly_score, thresholds
    except Exception as e:
        logger.error(f"모델 추론 중 오류 발생: {e}")
        raise

# -----------------------------------------------------------
# 요청 데이터 검증 함수
# -----------------------------------------------------------
def validate_input(req_data, sensor_type):
    """
    센서 유형에 따라 요청 데이터를 검증합니다.

    Args:
        req_data (dict): 들어오는 요청 데이터.
        sensor_type (str): 센서의 유형.

    Returns:
        tuple: (is_valid (bool), error_message (str))
    """
    required_fields = []
    if sensor_type == "Tilt Sensor":
        required_fields = ["value1", "value2", "degreeXAmount", "degreeYAmount", "temperature", "humidity"]
    elif sensor_type == "Crack Sensor":
        required_fields = ["crackAmount", "crackAmount2", "temperature", "humidity"]
    else:
        return False, "지원하지 않는 센서 유형입니다."

    missing_fields = [field for field in required_fields if field not in req_data["payload"]]
    if missing_fields:
        return False, f"payload에 누락된 필드가 있습니다: {', '.join(missing_fields)}"

    return True, ""

# -----------------------------------------------------------
# 추론 API 엔드포인트
# -----------------------------------------------------------
@app.route("/v1/inference/sensor-check", methods=["POST"])
def inference_endpoint():
    """
    센서 데이터를 기반으로 추론을 수행하는 API 엔드포인트.

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
                "status": True or False,
                "anomaly_scores": { ... },  # 실제 값
                "thresholds": { ... },      # 실제 값
                "prediction": { ... }       # 실제 값
            },
            "model_info": {
                "version": "Model_Name",
                "inference_time_ms": 123.45
            }
        }
    }
    """
    start_time = time.time()
    req_data = request.get_json()

    if req_data is None:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": "유효한 JSON 데이터를 제공해주세요."}}), 400

    if "deviceId" not in req_data:
        return jsonify({"error": {"code": "MISSING_FIELD", "message": "'deviceId' 필드가 누락되었습니다."}}), 400

    if "sensor_type" not in req_data:
        return jsonify({"error": {"code": "MISSING_FIELD", "message": "'sensor_type' 필드가 누락되었습니다."}}), 400

    device_id = req_data["deviceId"]
    sensor_type = req_data["sensor_type"]

    try:
        sensor_info = sensor_manager.get_model(device_id, sensor_type)
        sensor_type = sensor_info["sensor_info"]["type"]
        model_version = sensor_info["sensor_info"]["model_version"]
        model_path = sensor_info["sensor_info"]["model_path"]
        model_basename = os.path.splitext(os.path.basename(model_path))[0]  # 모델 경로의 베이스네임 추출
    except ValueError as ve:
        return jsonify({"error": {"code": "INVALID_SENSOR_TYPE", "message": str(ve)}}), 400
    except Exception:
        return jsonify({"error": {"code": "SERVER_ERROR", "message": "내부 서버 오류가 발생했습니다."}}), 500

    # 요청 데이터 검증
    valid, err_msg = validate_input(req_data, sensor_type)
    if not valid:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": err_msg}}), 400

    payload = req_data["payload"]
    timestamp_value = req_data["@timestamp"]

    # 센서 유형에 따른 센서 값 추출
    if sensor_type == "Tilt Sensor":
        sensor_values = {
            "value1": payload.get("value1"),
            "value2": payload.get("value2"),
            "degreeXAmount": payload.get("degreeXAmount"),
            "degreeYAmount": payload.get("degreeYAmount"),
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity")
        }
        max_vals = max_slope
        min_vals = min_slope
    elif sensor_type == "Crack Sensor":
        sensor_values = {
            "crackAmount": payload.get("crackAmount"),
            "crackAmount2": payload.get("crackAmount2"),
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity")
        }
        max_vals = max_crack
        min_vals = min_crack
    else:
        sensor_values = {}
        max_vals = None
        min_vals = None

    # 모델 추론 수행
    try:
        status, prediction, anomaly_scores, thresholds = model_inference(device_id, sensor_values, timestamp_value)
    except Exception:
        return jsonify({"error": {"code": "INFERENCE_ERROR", "message": "추론 중 오류가 발생했습니다."}}), 500

    anomaly_scores =  anomaly_scores.tolist() if isinstance(anomaly_scores, np.ndarray) else anomaly_scores
    thresholds = thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds
    prediction = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    keys = ['value1', 'value2', 'degreeXAmount', 'degreeYAmount', 'temperature', 'humidity', 'hour', 'day_of_week', 'month'] if len(prediction) == 9 \
        else ['crackAmount', 'crackAmount2', 'temperature', 'humidity', 'hour', 'day_of_week', 'month']

    # 정규화 해제
    if max_vals is not None and min_vals is not None:
        anomaly_scores_denorm = denormalize(anomaly_scores[0], max_vals, min_vals)
        thresholds_denorm = denormalize(thresholds[0], max_vals, min_vals)
        prediction_denorm = denormalize(prediction, max_vals, min_vals)
    else:
        anomaly_scores_denorm = anomaly_scores.tolist()
        thresholds_denorm = thresholds.tolist()
        prediction_denorm = prediction.tolist()

    # 추론 시간 측정 (ms)
    inference_time_ms = (time.time() - start_time) * 1000.0

    # 응답 JSON 구성
    keys_filtered = [key for key in keys if key not in ['hour', 'day_of_week', 'month']]
    response = {
        "data": {
            "timestamp": timestamp_value,
            "inference_result": {
                "status": str(status),
                "anomaly_scores": dict(zip(keys_filtered, anomaly_scores_denorm)),
                "thresholds": dict(zip(keys_filtered, thresholds_denorm)),
                "prediction": dict(zip(keys_filtered, prediction_denorm))
            },
            "model_info": {
                "model_name": model_basename,
                "inference_time_ms": round(inference_time_ms, 2)
            }
        }
    }
    logger.info(f"응답 생성됨: {response}")
    return jsonify(response), 200

# -----------------------------------------------------------
# window_months 설정을 읽거나 수정하는 API 엔드포인트
# -----------------------------------------------------------
@app.route("/v1/config/window_months", methods=["GET", "POST"])
def manage_window_months():
    """
    window_months 설정을 읽거나 수정하는 API 엔드포인트.

    GET 요청:
        - 현재 window_months 값을 반환합니다.

    POST 요청:
        - JSON 본문에 "window_months" 필드를 포함하여 값을 수정합니다.
        - 예시 JSON 본문:
          {
              "window_months": 6
          }

    Response Format:
        {
            "window_months": 현재 설정된 값
        }
    """
    if request.method == "GET":
        return jsonify({"window_months": threshold_manager.window_months}), 200

    elif request.method == "POST":
        req_data = request.get_json()
        if req_data is None:
            return jsonify({"error": {"code": "INVALID_INPUT", "message": "유효한 JSON 데이터를 제공해주세요."}}), 400

        if "window_months" not in req_data:
            return jsonify({"error": {"code": "MISSING_FIELD", "message": "'window_months' 필드가 필요합니다."}}), 400

        new_window = req_data["window_months"]

        if not isinstance(new_window, int) or new_window <= 0:
            return jsonify({"error": {"code": "INVALID_VALUE", "message": "'window_months'는 양의 정수여야 합니다."}}), 400

        try:
            threshold_manager.window_months = new_window
            threshold_manager.prune_history()  # 새로운 기간에 맞게 히스토리 정리
            threshold_manager.save_history()    # 업데이트된 설정 저장
            logger.info(f"window_months가 {new_window}으로 업데이트되었습니다.")
            return jsonify({"window_months": new_window}), 200
        except Exception as e:
            logger.error(f"window_months 업데이트 중 오류 발생: {e}")
            return jsonify({"error": {"code": "SERVER_ERROR", "message": "설정 업데이트 중 오류가 발생했습니다."}}), 500

# -----------------------------------------------------------
# Flask 서버 실행 함수
# -----------------------------------------------------------
def run_server():
    """
    Flask 서버를 실행하는 함수.
    """
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # werkzeug의 로그 레벨을 ERROR로 설정하여 WARNING 메시지를 숨깁니다.

    try:
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 8080))
        debug = os.getenv('FLASK_DEBUG', 'False') == 'True'  # 기본적으로 디버그 모드를 비활성화합니다.

        app.run(host=host, port=port, debug=debug, use_reloader=True)  # 자동 리로더 활성화
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_server()
