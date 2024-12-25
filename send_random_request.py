import os
import csv
import requests
import time
from datetime import datetime

API_URL = "http://localhost:5000/v1/inference/sensor-check"
CSV_DIRECTORY = "/workspace/data/github/TranAD/valdata"  # CSV 파일들이 있는 디렉토리 경로

def send_sequential_request():
    """
    CSV 디렉토리의 파일들을 'yyyymm' 순으로 정렬하고,
    각 파일 내의 데이터를 timestamp 기준으로 정렬하여 API에 요청을 보냅니다.
    각 요청은 5초 간격으로 전송됩니다.
    """
    # CSV 파일 리스트를 'yyyymm' 순으로 정렬된 순서로 가져오기
    try:
        csv_files = sorted(
            [file for file in os.listdir(CSV_DIRECTORY) if file.endswith('.csv')],
            key=lambda x: datetime.strptime(x[:6], "%Y%m")
        )
        if not csv_files:
            print("CSV 파일을 찾을 수 없습니다.")
            return
    except FileNotFoundError:
        print(f"디렉토리를 찾을 수 없습니다: {CSV_DIRECTORY}")
        return
    except ValueError as ve:
        print(f"파일 이름 형식 오류: {ve}")
        return

    for csv_file in csv_files:
        file_path = os.path.join(CSV_DIRECTORY, csv_file)
        print(f"처리 중인 파일: {csv_file}")

        # CSV 파일 읽기
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter='|')
                header = next(reader, None)
                if header is None:
                    print(f"{csv_file}에 데이터가 없습니다.")
                    continue

                # 모든 데이터 행을 리스트로 읽고 timestamp 기준으로 정렬
                data_rows = []
                for idx, data_row in enumerate(reader, start=1):
                    if len(data_row) < 3:
                        print(f"{csv_file}의 {idx}번째 행에 데이터가 충분하지 않습니다.")
                        continue

                    # 첫 번째 열은 ID, 두 번째 열은 timestamp
                    id_str = data_row[0]
                    timestamp_str = data_row[1]
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError as ve:
                        print(f"타임스탬프 형식 오류 ({csv_file}의 {idx}번째 행): {ve}")
                        continue

                    try:
                        # 센서 데이터는 세 번째 열부터 시작한다고 가정
                        sensor_data = {f"sensor_{i+1}": float(value) for i, value in enumerate(data_row[2:], start=0)}
                    except ValueError as ve:
                        print(f"센서 데이터 변환 오류 ({csv_file}의 {idx}번째 행): {ve}")
                        continue

                    data_rows.append((timestamp, sensor_data))

                # timestamp 기준으로 데이터 정렬
                sorted_data = sorted(data_rows, key=lambda x: x[0])

                for idx, (timestamp, sensor_data) in enumerate(sorted_data, start=1):
                    # 요청 데이터 구성
                    payload = {
                        "data": {
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "sensor_data": sensor_data
                        }
                    }

                    # POST 요청 보내기
                    try:
                        response = requests.post(API_URL, json=payload)
                        if response.status_code == 200:
                            print(f"{csv_file}의 {idx}번째 행 전송 성공: {response.json()}")
                        else:
                            print(f"{csv_file}의 {idx}번째 행 전송 오류: {response.status_code}, {response.text}")
                    except Exception as e:
                        print(f"{csv_file}의 {idx}번째 행 전송 중 예외 발생: {e}")

                    # 5초 대기
                    time.sleep(0.1)

        except Exception as e:
            print(f"{csv_file} 파일을 읽는 중 오류 발생: {e}")

if __name__ == "__main__":
    send_sequential_request()
