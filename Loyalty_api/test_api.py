import requests

url = 'http://127.0.0.1:5001/predict'

input_data = {
    "input_data": [3000]  
}

try:
    response = requests.post(url, json=input_data)

    if response.status_code == 200:
        print("Prediction result:", response.json())
    else:
        print("Error:", response.status_code, response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
