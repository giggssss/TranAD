# api_server.py
# REST API 서버 예시 코드
# 이 코드는 TranAD_Gyeongsan 모델 체크포인트를 기반으로 Inference를 수행하는 REST-API 엔드포인트 구현 예제임.

from flask import Flask, request, jsonify
from datetime import datetime
import time
import re
import torch
import numpy as np
import os
import sys
import logging
import src.dataprocessor as dp
import json
import torch.nn as nn
from collections import deque
from datetime import datetime, timedelta

# 모델 클래스 임포트
import src.models   # 필요한 모델 추가
import src.utils    # 센서 데이터 전처리 함수가 있다고 가정

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# DataProcessor 인스턴스 생성 (필요에 따라 초기화 매개변수 추가)
data_processor = dp.DataProcessor()

# -----------------------------------------------------------
# ThresholdManager 클래스 추가
# -----------------------------------------------------------
class ThresholdManager:
    """
    최근 설��된 기간(기본: 1년) 동안의 anomaly_scores를 누적하여 thresholds를 계산하는 클래스.
    사용자 옵션으로 최근 3개월 또는 1년을 기준으로 선택할 수 있습니다.
    """
    def __init__(self, window_months=12, storage_path="anomaly_scores.json"):
        """
        초기화 함수.
        
        Args:
            window_months (int): 히스토리를 유지할 월 수 (기본값: 12개월).
            storage_path (str): anomaly_scores를 저장할 파일 경로.
        """
        self.window_months = window_months
        self.storage_path = storage_path
        self.anomaly_history = deque()
        self.load_history()

    def load_history(self):
        """
        저장된 anomaly_scores 히스토리를 로드합니다.
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
                logger.info(f"히스토리에서 {len(self.anomaly_history)}개의 anomaly scores 로드 완료.")
            except Exception as e:
                logger.error(f"히스토리 로드 중 오류 발생: {e}")
                self.anomaly_history = deque()
        else:
            self.anomaly_history = deque()
            logger.info("히스토리 파일이 존재하지 않아서 새로 생성됩니다.")

    def save_history(self):
        """
        현재의 anomaly_scores 히스토리를 저장합니다.
        """
        try:
            # anomaly_history가 비어있더라도 현재 상태를 저장합니다.
            data = [
                {
                    "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "anomaly_scores": entry["scores"]
                }
                for entry in self.anomaly_history
            ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"{len(data)}개의 anomaly scores를 히스토리에 저장 완료.")
        except Exception as e:
            logger.error(f"히스토리 저장 중 오류 발생: {e}")

    def prune_history(self):
        """
        설정된 기간 내의 데이터만 유지하도��� 히스토리를 정리합니다.
        """
        if self.anomaly_history:
            latest_timestamp = max(entry["timestamp"] for entry in self.anomaly_history)
            cutoff_date = latest_timestamp - timedelta(days=self.window_months * 30)  # 대략적인 월수 계산
        else:
            cutoff_date = datetime.now() - timedelta(days=self.window_months * 30)  # 대략적인 월수 계산
        initial_length = len(self.anomaly_history)
        while self.anomaly_history and self.anomaly_history[0]["timestamp"] < cutoff_date:
            self.anomaly_history.popleft()
        pruned_length = len(self.anomaly_history)
        logger.info(f"히스토리 정리: {initial_length}개에서 {pruned_length}개로 감소.")

    def add_anomaly_scores(self, timestamp, scores):
        """
        새로운 anomaly_scores를 히스토리에 추가합니다.
        
        Args:
            timestamp (datetime): 데이터의 타임스탬프.
            scores (list): anomaly_scores 리스트.
        """
        self.anomaly_history.append({"timestamp": timestamp, "scores": scores})
        logger.info(f"Anomaly scores 추가: {timestamp} - {scores}")
        self.prune_history()
        self.save_history()

    def calculate_thresholds(self, current_scores):
        """
        설정된 ���간 동안의 anomaly_scores를 기반으로 thresholds를 계산합니다.
        데이터가 없을 경우 현재의 scores를 사용합니다.

        Args:
            current_scores (numpy.ndarray): 현재의 anomaly_scores

        Returns:
            numpy.ndarray: 계산된 thresholds
        """
        if not self.anomaly_history:
            logger.warning("히스토리에 데이터가 없어 현재의 anomaly_scores를 기준으로 thresholds를 계산합니다.")
            mean_scores = np.mean(current_scores, axis=0)
            std_scores = np.std(current_scores, axis=0)
        else:
            all_scores = np.array([entry["scores"] for entry in self.anomaly_history])
            mean_scores = np.mean(all_scores, axis=0)
            std_scores = np.std(all_scores, axis=0)
        thresholds = mean_scores + 3 * std_scores
        logger.info(f"계산된 thresholds: {thresholds}")
        return thresholds

# -----------------------------------------------------------
# ThresholdManager 인스턴스 초기화 (절대 경로 사용 예시)
# -----------------------------------------------------------
threshold_manager = ThresholdManager(
    window_months=12,  # 기본 기간을 12개월로 설정
    storage_path=os.path.join(os.path.dirname(__file__), "anomaly_scores.json")
)

# -----------------------------------------------------------
# 모델 로딩 함수
# -----------------------------------------------------------
def load_model(model_name, model_path, input_dim):
    """
    실제 모델을 로딩하는 함수.

    Args:
        model_name (str): 로딩할 모델의 이름.
        model_path (str): 모델 체크포인트 파일 경로.
        input_dim (int): 입력 데이터의 차원.

    Returns:
        torch.nn.Module: 로드된 모델.
    """    
    try:
        model_class = getattr(src.models, model_name)
    except AttributeError:
        logger.error(f"모델 클래스 '{model_name}'을(를) src.models에서 찾을 수 없습니다.")
        raise

    model = model_class(input_dim).double()
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"{model_name} 모델이 성공적으로 로딩되었습니다.")
    except Exception as e:
        logger.error(f"모델 체크포인트 로딩 중 오류 발생: {e}")
        raise
    return model

# -----------------------------------------------------------
# 모델 초기화 (서버 시작 시 한 번만 실행)
# -----------------------------------------------------------
def initialize_model():
    MODEL_NAME = "TranAD"
    MODEL_VERSION = "v1.0" # 사용하려는 모델 version
    MODEL_PATH = "/workspace/checkpoints/TranAD_Gyeongsan/model.ckpt"
    INPUT_DIM = 13  # 센서 데이터의 차원 (10개 센서 + 3개 시간 관련 특성)

    try:
        model = load_model(MODEL_NAME, MODEL_PATH, INPUT_DIM)
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        sys.exit(1)
    
    return model, MODEL_NAME, INPUT_DIM, MODEL_VERSION

# 모델과 관련 정보 전역 변수로 초기화
model, MODEL_NAME, INPUT_DIM, MODEL_VERSION = initialize_model()

# -----------------------------------------------------------
# timestamp 형식 검증 함수 (YYYY-MM-DD HH:MM:SS 형태)
# -----------------------------------------------------------
def validate_timestamp(ts):
    """
    timestamp가 'YYYY-MM-DD HH:MM:SS' 형식인지 검증하는 함수.

    Args:
        ts (str): 검증할 timestamp 문자열.

    Returns:
        bool: 유효한 형식이면 True, 아니면 False.
    """
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    return bool(re.match(pattern, ts))

# -----------------------------------------------------------
# 입력 데이터 검증 함수
# -----------------------------------------------------------
def validate_input(data, input_dim):
    """
    입력 데이터의 유효성을 검증하는 함수.

    Args:
        data (dict): 검증할 입력 데이터.
        input_dim (int): 센서 데이터의 예상 개수 (10).

    Returns:
        tuple: (유효 여부 (bool), 오류 메시지 (str 또는 None))
    """
    if "data" not in data:
        return False, "Missing 'data' field"

    input_data = data["data"]
    
    # timestamp 필드 확인
    if "timestamp" not in input_data:
        return False, "timestamp field가 없습니다."
    if not validate_timestamp(input_data["timestamp"]):
        return False, "timestamp 필드가 없거나 형식이 잘못되었습니다. 'YYYY-MM-DD HH:MM:SS' 형식을 사용해주세요."
    
    # sensor_data 필드 확인 및 센서 개수 검증
    if "sensor_data" not in input_data:
        return False, "sensor_data field가 없습니다."
    
    sensor_data = input_data["sensor_data"]
    if not isinstance(sensor_data, dict):
        return False, "sensor_data 필드는 사전(dictionary) 형식이어야 합니다."
    if len(sensor_data.keys()) != input_dim:
        return False, f"sensor_data는 정확히 {input_dim}개의 센서 값을 포함해야 합니다."
    
    # 각 센서 값이 float 형식인지 확인
    for k, v in sensor_data.items():
        if not isinstance(v, (float, int)):
            return False, f"sensor {k} 는 float 또는 int 형식이어야 합니다."
    
    return True, None

# -----------------------------------------------------------
# 전처리 함수 정의
# -----------------------------------------------------------
def preprocess_sensor_data(sensor_values, timestamp, input_dim):
    """
    센서 데이터를 전처리하는 함수.

    Args:
        sensor_values (list): 센서 값 리스트 (10개).
        timestamp (str): 타임스탬프 문자열.
        input_dim (int): 센서 데이터의 차원 (10).

    Returns:
        tuple: 전처리된 window 텐서과 elem 텐서.
    """
    try:
        # 타임스탬프를 datetime 객체로 변환
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
    except ValueError:
        logger.error("타임스탬프 형식이 잘못되었습니다.")
        raise

    # 센서 데이터에 시간 관련 특성 추가
    sensor_values_extended = sensor_values + [hour, day_of_week, month]

    # 센서 데이터를 NumPy 배열로 변환
    sensor_array = np.array(sensor_values_extended, dtype=np.float64)

    # 데이터 정규화
    try:
        normalized_data, _, _ = data_processor.normalize3(sensor_array)
    except Exception as e:
        logger.error(f"데이터 정규화 중 오류 발생: {e}")
        raise

    # 정규화된 데이터를 텐서로 변환하고 반복
    sensor_tensor = torch.from_numpy(normalized_data).double().unsqueeze(0)  # Shape: [1, 13]
    sensor_tensor = sensor_tensor.expand(10, -1)  # Shape: [10, 13]

    window = sensor_tensor.unsqueeze(1)  # Shape: [10, 1, 13]
    elem = window[-1, :, :].view(1, 1, input_dim)  # Shape: [1, 1, 13]

    return window, elem

# -----------------------------------------------------------
# 추론 함수
# -----------------------------------------------------------
def model_inference(model, sensor_values, timestamp):
    """
    실제 모델을 사용하여 추론을 수행하는 함수.
    
    Args:
        model (torch.nn.Module): 로드된 모델.
        sensor_values (list): 센서 값 리스트 (10개).
        timestamp (str): 타임스탬프 문자열.
    
    Returns:
        tuple: (status, anomaly_scores, thresholds)
    """
    try:
        # 센서 데이터를 텐서로 변환 및 전처리
        window, elem = preprocess_sensor_data(sensor_values, timestamp, INPUT_DIM)  # 전처리 함수 사용
    except Exception as e:
        logger.error(f"전처리 중 오류 발생: {e}")
        return "False", np.array([0.0]*INPUT_DIM), np.array([0.0]*INPUT_DIM), np.array([0.0]*INPUT_DIM)

    try:
        l = nn.MSELoss(reduction = 'none')
        
        with torch.no_grad():
            output = model(window, elem)
            if isinstance(output, tuple):
                output = output[1]
            anomlay_score = l(output, elem)[0].cpu().numpy()
            
            # 모델의 출력이 여러 차원의 이상치 점수일 경우를 가정
            prediction = output.cpu().numpy().flatten()  # [num_features]
        logger.info(f"모델 추론 완료: {prediction}")
    except Exception as e:
        logger.error(f"모델 추론 중 오류 발생: {e}")
        return "False", np.array([0.0]*INPUT_DIM), np.array([0.0]*INPUT_DIM)

    try:
        # 현재 anomaly_scores를 히스토리에 추가
        timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        threshold_manager.add_anomaly_scores(timestamp_dt, anomlay_score.tolist())

        # 최근 3개월간의 데이터를 기반으로 thresholds 계산
        thresholds = threshold_manager.calculate_thresholds(anomlay_score)

        # 각 특성별로 임계값을 초과하는 이상치 인덱스 찾기
        anomalies = np.where(anomlay_score > thresholds)[0]

        # 이상치가 하나라도 존재하면 status를 "True"로 설정
        status = "True" if anomalies.size > 0 else "False"

        logger.info(f"추론 결과: status={status}, anomalies={anomalies}")
        return status, prediction, anomlay_score, thresholds
    except Exception as e:
        logger.error(f"threshold 계산 중 오류 발생: {e}")
        return "False", prediction, anomlay_score, np.array([0.0]*INPUT_DIM)

# -----------------------------------------------------------
# API 엔드포인트 구현
# -----------------------------------------------------------
@app.route("/v1/inference/sensor-check", methods=["POST"])
def inference_endpoint():
    """
    센서 데이터 기반 추론을 수행하는 API 엔드포인트.
    
    요청 형식:
    {
        "data": {
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "sensor_data": {
                "sensor_1": float,
                "sensor_2": float,
                ...
                "sensor_10": float
            }
        }
    }
    
    응답 형식:
    {
        "data": {
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "inference_result": {
                "status": "True" 또는 "False",
                "anomaly_scores": [float, ...],
                "thresholds": [float, ...]
            },
            "model_info": {
                "version": "모델 버전",
                "inference_time_ms": float
            }
        }
    }
    """
    start_time = time.time()
    req_data = request.get_json()
    
    if req_data is None:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": "유효한 JSON 데이터를 제공해주세요."}}), 400

    # 요청 데이터 검증
    valid, err_msg = validate_input(req_data, 10)  # 센서 데이터는 10개
    if not valid:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": err_msg}}), 400
    
    input_data = req_data["data"]
    timestamp_value = input_data["timestamp"]
    sensor_values = list(input_data["sensor_data"].values())

    # 모델 추론 수행
    status, prediction, anomaly_scores, thresholds = model_inference(model, sensor_values, timestamp_value)

    # 추론 시간 측정(ms)
    inference_time_ms = (time.time() - start_time) * 1000.0

    # 응답 JSON 구성
    response = {
        "data": {
            "timestamp": timestamp_value,
            "inference_result": {
                "status": status,
                "anomaly_scores": anomaly_scores.tolist() if isinstance(anomaly_scores, np.ndarray) else anomaly_scores,
                "thresholds": thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds,
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
            },
            "model_info": {
                "version": f"{MODEL_NAME} {MODEL_VERSION}",
                "inference_time_ms": round(inference_time_ms, 2)
            }
        }
    }
    logger.info(f"응답 생성 완료: {response}")
    return jsonify(response), 200

# -----------------------------------------------------------
# window_months 설정을 읽고 수정할 수 있는 API 엔드포인트 추가
# -----------------------------------------------------------
@app.route("/v1/config/window_months", methods=["GET", "POST"])
def manage_window_months():
    """
    window_months 설정을 읽거나 수정하는 API 엔드포인트.
    
    GET 요청:
        - 현재 설정된 window_months 값을 반환합니다.
    
    POST 요청:
        - JSON 본문에 "window_months" 필드를 포함하여 값을 수정합니다.
        - 예시 요청 본문:
          {
              "window_months": 6
          }
    
    응답 형식:
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
            threshold_manager.save_history()   # 변경된 설정 저장
            logger.info(f"window_months가 {new_window}으로 변경되었습니다.")
            return jsonify({"window_months": new_window}), 200
        except Exception as e:
            logger.error(f"window_months 변경 중 오류 발생: {e}")
            return jsonify({"error": {"code": "SERVER_ERROR", "message": "설정을 변경하는 중 오류가 발생했습니다."}}), 500

# -----------------------------------------------------------
# 메인 실행부 수정
# -----------------------------------------------------------
def run_server():
    """
    Flask 서버를 실행하는 함수.
    """
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # werkzeug의 로그 레벨을 ERROR로 설정하여 WARNING 메시지 숨김

    try:
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 8080))
        debug = os.getenv('FLASK_DEBUG', 'False') == 'True'  # 디버그 모드를 기본적으로 True로 설정

        app.run(host=host, port=port, debug=debug, use_reloader=True)  # 자동 리로더 활성화
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_server()
