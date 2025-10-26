import requests
import json

print("--- Testing ML API ---")

# Test home
r1 = requests.get('http://localhost:5000/')
print(f"Home: {r1.status_code} -> {r1.text}")

# Test valid prediction
data = {"features": [5.5, 3.2, 8.1, 1.9, 6.7]}
r2 = requests.post('http://localhost:5000/predict', headers={'Content-Type':'application/json'}, data=json.dumps(data))
print(f"Valid Prediction: {r2.status_code} -> {r2.json()}")

# Test invalid feature count
data_invalid = {"features": [1.0, 2.0]}
r3 = requests.post('http://localhost:5000/predict', headers={'Content-Type':'application/json'}, data=json.dumps(data_invalid))
print(f"Invalid Features: {r3.status_code} -> {r3.json()}")

# Test non-JSON input
r4 = requests.post('http://localhost:5000/predict', data="Not JSON")
print(f"Invalid JSON: {r4.status_code} -> {r4.text}")
