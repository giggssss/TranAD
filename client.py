import requests
import json
import base64
import random

server_ip = "127.0.0.1"
server_port = 8080

def send_inference_request(device_id, sensor_type, sensor_payload, timestamp):
    """
    Sends an inference request to the server based on sensor data.
    
    Args:
        device_id (str): Unique ID of the sensor.
        sensor_type (str): Type of the sensor.
        sensor_payload (dict): Sensor data payload.
        timestamp (str): Timestamp string ('YYYY-MM-DD HH:MM:SS').
        
    Example:
        device_id = "sensor_01"
        sensor_type = "Tilt Sensor"
        sensor_payload = {
            "cnt": 2,
            "rssi": -55,
            "seqno": "172",
            "idx": 0,
            "intervalTimeSet": 60,
            "batLevel": 80,
            "temperature": -1,
            "humidity": 47,
            "value1": -0.6,
            "value2": -2.69,
            "degreeXAmount": 0.6,
            "degreeYAmount": 2.69
        }
        timestamp = "2024-12-24 00:00:00"
    """
    url = f"http://{server_ip}:{server_port}/v1/inference/sensor-check"
    data = {
        "deviceId": device_id,
        "sensor_type": sensor_type,
        "payload": sensor_payload,
        "@timestamp": timestamp
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        try:
            # Decode the response content as UTF-8
            decoded_content = response.content.decode('utf-8')
            response_json = json.loads(decoded_content)
            
            # Assume certain fields in the response are Base64 encoded
            if 'encoded_data' in response_json:
                encoded_data = response_json['encoded_data']
                # Decode Base64
                decoded_data = base64.b64decode(encoded_data).decode('utf-8')
                response_json['decoded_data'] = decoded_data
                del response_json['encoded_data']
            
            # Print the final response
            print("Server Response:", json.dumps(response_json, ensure_ascii=False, indent=4))
        except (json.JSONDecodeError, UnicodeDecodeError, base64.binascii.Error) as e:
            print("Response Decoding Error:", str(e))
    else:
        print("Error Occurred:", response.status_code, response.text)

def get_window_months():
    """
    Retrieves the current window_months value from the server.
    """
    url = f"http://{server_ip}:{server_port}/v1/config/window_months"
    response = requests.get(url)
    if response.status_code == 200:
        print("Current window_months:", json.dumps(response.json(), ensure_ascii=False, indent=4))
    else:
        print("Error Occurred:", response.status_code, response.text)

def change_window_months(new_window_months):
    """
    Changes the window_months value on the server.
    
    Args:
        new_window_months (int): New window_months value to set.
        
    Example:
        new_window_months = 6
    """
    url = f"http://{server_ip}:{server_port}/v1/config/window_months"
    data = {"window_months": new_window_months}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("window_months Updated:", json.dumps(response.json(), ensure_ascii=False, indent=4))
    else:
        print("Error Occurred:", response.status_code, response.text)

def send_tilt_sensor_request():
    """
    Sends example tilt sensor data to the server.
    """
    device_id = "sensor_01"
    sensor_type = "Tilt Sensor"
    sensor_payload = {
        "cnt": 2,
        "rssi": random.randint(-100, 0),
        "seqno": str(random.randint(100, 200)),
        "idx": random.randint(0, 10),
        "intervalTimeSet": random.randint(30, 120),
        "batLevel": random.randint(0, 100),
        "temperature": random.uniform(-20, 40),
        "humidity": random.uniform(0, 100),
        "value1": random.uniform(-5, 5),
        "value2": random.uniform(-5, 5),
        "degreeXAmount": random.uniform(0, 10),
        "degreeYAmount": random.uniform(0, 10)
    }
    timestamp = "2024-12-24 00:00:00"
    send_inference_request(device_id, sensor_type, sensor_payload, timestamp)

def send_crack_sensor_request():
    """
    Sends example crack sensor data to the server.
    """
    device_id = "sensor_02"
    sensor_type = "Crack Sensor"
    sensor_payload = {
        "cnt": 2,
        "rssi": random.randint(-100, 0),
        "seqno": str(random.randint(100, 200)),
        "idx": random.randint(0, 10),
        "intervalTimeSet": random.randint(30, 120),
        "batLevel": random.randint(0, 100),
        "temperature": random.uniform(-20, 40),
        "humidity": random.uniform(0, 100),
        "crackAmount": random.uniform(0, 1),
        "crackAmount2": random.uniform(0, 1)
    }
    timestamp = "2024-12-24 00:00:00"
    send_inference_request(device_id, sensor_type, sensor_payload, timestamp)

if __name__ == "__main__":
    # # Retrieve current window_months
    # get_window_months()
    
    # # Change window_months to 6
    # change_window_months(6)
    
    # # Retrieve updated window_months
    # get_window_months()
    for i in range(1):
        # Send inference request for tilt sensor
        send_tilt_sensor_request()
        
        # Send inference request for crack sensor
        send_crack_sensor_request()