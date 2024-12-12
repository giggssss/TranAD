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

# 모델 클래스 임포트
import src.models   # 필요한 모델 추가
import src.utils    # 센서 데이터 전처리 함수가 있다고 가정

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DataProcessor 인스턴스 생성 (필요에 따라 초기화 매개변수 추가)
data_processor = dp.DataProcessor()

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
    model_class = getattr(src.models, model_name)
    model = model_class(input_dim).double()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# -----------------------------------------------------------
# 모델 초기화 (서버 시작 시 한 번만 실행)
# -----------------------------------------------------------
def initialize_model():
    MODEL_NAME = "TranAD"
    MODEL_VERSION = "v1.0" # 사용하려는 모델 version
    MODEL_PATH = "/workspace/data/github/TranAD/checkpoints/TranAD_Gyeongsan/model.ckpt"
    INPUT_DIM = 13  # 센서 데이터의 차원 (10개 센서 + 3개 시간 관련 특성)

    try:
        model = load_model(MODEL_NAME, MODEL_PATH, INPUT_DIM)
        logger.info(f"{MODEL_NAME} 모델이 성공적으로 로딩되었습니다.")
    except Exception as e:
        logger.error(f"모델 로딩 중 오류 발생: {e}")
        sys.exit(1)
    
    return model, MODEL_NAME, INPUT_DIM, MODEL_VERSION


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
            return False, f"sensor {k} 값은 float 또는 int 형식이어야 합니다."
    
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
    normalized_data, _, _ = data_processor.normalize3(sensor_array)

    # 정규화된 데이터를 텐서로 변환하고 반복
    sensor_tensor = torch.from_numpy(normalized_data).double().tile(1, 10, 1)  # Shape: [1, 10, 13]

    local_bs = sensor_tensor.shape[0]
    window = sensor_tensor.permute(1, 0, 2)
    elem = window[-1, :, :].view(1, local_bs, input_dim)

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
        tuple: (status, anomaly_scores)
    """
    def smooth(y, box_pts=1):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    try:
        # 센서 데이터를 텐서로 변환 및 전처리
        window, elem = preprocess_sensor_data(sensor_values, timestamp, 13)  # 전처리 함수 사용
        with torch.no_grad():
            output = model(window, elem)
            if isinstance(output, tuple): output = output[1]
            # 모델의 출력이 여러 차원의 이상치 점수일 경우를 가정
            anomaly_scores = output.numpy()  # [num_samples, num_features]

        # 각 특성별 평균과 표준편차 계산
        mean_scores = np.mean(smooth(anomaly_scores), axis=0)
        std_scores = np.std(smooth(anomaly_scores), axis=0)
        thresholds = mean_scores + 3 * std_scores

        # 각 특성별로 임계값을 초과하는 이상치 인덱스 찾기
        anomalies = np.where(anomaly_scores > thresholds)[0]

        # 이상치가 하나라도 존재하면 status를 "True"로 설정
        status = "True" if anomalies.size > 0 else "False"

        return status, anomaly_scores, thresholds
    except Exception as e:
        logger.error(f"추론 중 오류 발생: {e}")
        return "False", 0.0, 0.0 

# -----------------------------------------------------------
# API 엔드포인트 구현
# -----------------------------------------------------------
@app.route("/v1/inference/sensor-check", methods=["POST"])
def inference_endpoint():
    """
    센서 데이터 기반 추론을 수행하는 API ��드포인트.
    
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
                "anomaly_scores": [float, ...]
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
    
    # 요청 데이터 검증
    valid, err_msg = validate_input(req_data, 10)  # 센서 데이터는 10개
    if not valid:
        return jsonify({"error": {"code": "INVALID_INPUT", "message": err_msg}}), 400
    
    input_data = req_data["data"]
    timestamp_value = input_data["timestamp"]
    sensor_values = list(input_data["sensor_data"].values())

    # 모델 추론 수행
    status, anomaly_scores, thresholds = model_inference(model, sensor_values, timestamp_value)

    # 추론 시간 측정(ms)
    inference_time_ms = (time.time() - start_time) * 1000.0

    # 응답 JSON 구성
    response = {
        "data": {
            "timestamp": timestamp_value,
            "inference_result": {
                "status": status,
                "anomaly_scores": anomaly_scores.tolist() if isinstance(anomaly_scores, np.ndarray) else anomaly_scores,
                "thresholds": thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds
            },
            "model_info": {
                "version": f"{MODEL_NAME} {MODEL_VERSION}",
                "inference_time_ms": round(inference_time_ms, 2)
            }
        }
    }
    return jsonify(response), 200

# -----------------------------------------------------------
# 메인 실행부
# -----------------------------------------------------------
# 모델과 관련 정보 전역 변수로 초기화
model, MODEL_NAME, INPUT_DIM, MODEL_VERSION = initialize_model()

def run_server():
    """
    Flask 서버를 실행하는 함수.
    """
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    run_server()