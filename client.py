import requests
import json

# API 서버 URL
url = "http://localhost:5000/v1/inference/sensor-check"

# 요청 데이터 예시
data = {
    "data": {
        "timestamp": "2024-01-01 00:00:00",
        "sensor_data": {
            "sensor_1": 23.5,
            "sensor_2": 24.0,
            "sensor_3": 22.8,
            "sensor_4": 23.1,
            "sensor_5": 10.3,
            "sensor_6": 23.9,
            "sensor_7": 24.1,
            "sensor_8": 22.7,
            "sensor_9": 23.4,
            "sensor_10": 24.2
        }
    }
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 처리
if response.status_code == 200:
    print("서버 응답:", json.dumps(response.json(), ensure_ascii=False, indent=4))
else:
    print("오류 발생:", response.status_code, response.text) 