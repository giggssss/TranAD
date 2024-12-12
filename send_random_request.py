import os
import random
import csv
import requests

API_URL = "http://localhost:5000/v1/inference/sensor-check"
CSV_DIRECTORY = "/workspace/data/github/TranAD/valdata"  # CSV 파일들이 있는 디렉토리 경로

def send_random_request():
    """
    CSV 디렉토리에서 랜덤 파일과 데이터를 선택하여 API에 요청을 보냅니다.
    """
    # CSV 파일 리스트 가져오기
    try:
        csv_files = [file for file in os.listdir(CSV_DIRECTORY) if file.endswith('.csv')]
        if not csv_files:
            print("CSV 파일을 찾을 수 없습니다.")
            return
    except FileNotFoundError:
        print(f"디렉토리를 찾을 수 없습니다: {CSV_DIRECTORY}")
        return

    # 랜덤으로 CSV 파일 선택
    selected_file = random.choice(csv_files)
    file_path = os.path.join(CSV_DIRECTORY, selected_file)
    print(f"선택된 파일: {selected_file}")

    # CSV 파일 읽기
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter='|'))
            if len(reader) < 2:
                print("선택한 CSV 파일에 데이터가 충분하지 않습니다.")
                return

            # 랜덤으로 데이터 행 선택 (헤더 제외)
            data_row = random.choice(reader[1:])
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류 발생: {e}")
        return

    # 타임스탬프와 센서 데이터 생성 (가정: 첫 열은 타임스탬프, 이후 열은 센서 데이터)
    timestamp = data_row[1]
    try:
        sensor_data = {f"sensor_{i+1}": float(value) for i, value in enumerate(data_row[2:], start=0)}
    except ValueError as ve:
        print(f"센서 데이터 변환 오류: {ve}")
        return

    # 요청 데이터 구성
    payload = {
        "data": {
            "timestamp": timestamp,
            "sensor_data": sensor_data
        }
    }

    # POST 요청 보내기
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            print("서버 응답:", response.json())
        else:
            print("오류 발생:", response.status_code, response.text)
    except Exception as e:
        print("요청 중 예외 발생:", e)

if __name__ == "__main__":
    for i in range(50):
        send_random_request() 
