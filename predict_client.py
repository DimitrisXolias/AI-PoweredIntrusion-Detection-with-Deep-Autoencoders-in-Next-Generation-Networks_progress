
import requests
import json

# Ensure this list has exactly 84 elements
features = [
    52462.0, 54992.0, 21.0, 1.0, 0.0,
    1e-06, 1.0, 1.0, 1.0, 0.0, 1048576.0, 1048576.0, 2097152.0, 1.0, 20.0, 20.0,
    20.0, 20.0, 20.0, 20.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 120.0, 120.0, 120.0, 120.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 120.0, 120.0, 60.0, 84.852814, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.953674, 0.953674, 0.953674, 0.953674, 0.0, 125829120.0, 1.0, 1.0, 120.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.953674, 0.953674, 0.953674, 0.953674, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 64.0, 0.0, 64.0
]

url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
data = json.dumps({'features': features})

try:
    response = requests.post(url, headers=headers, data=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except Exception as e:
    print("Other error:", e)

if response.status_code != 200:
    print("Server error response:", response.text)
