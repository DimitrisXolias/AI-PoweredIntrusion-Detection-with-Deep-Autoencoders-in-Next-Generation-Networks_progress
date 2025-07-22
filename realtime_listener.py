import pyshark
import os
import time
import pandas as pd
import numpy as np
import requests
import joblib
import datetime
import subprocess

# === PATHS ===
BASE_DIR = "/home/xold/ids-project"
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_raw83.pkl")
LOG_PATH = os.path.join(BASE_DIR, "data", "results.log")
INTERFACE = "r1-eth2"
API_URL = "http://127.0.0.1:5000/predict"

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(pkt):
    try:
        return [
            float(pkt.length),
            float(pkt.sniff_timestamp),
            pkt.highest_layer == "TCP",
            pkt.highest_layer == "UDP",
        ]
    except Exception:
        return None

# === CHECK IF INTERFACE EXISTS ===
def check_interface(interface):
    result = subprocess.run(["ip", "link", "show", interface],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return result.returncode == 0

# === MAIN ===
if __name__ == "__main__":
    print("📦 Loading scaler...")
    scaler = joblib.load(open(SCALER_PATH, "rb"))

    print(f"🎧 Listening in real time on {INTERFACE}...")

    if not check_interface(INTERFACE):
        print(f"❌ Interface '{INTERFACE}' not found. Is Mininet running?")
        exit(1)

    capture = pyshark.LiveCapture(interface=INTERFACE)
    
    for pkt in capture.sniff_continuously():
        features = extract_features(pkt)
        if features is None:
            continue

        df = pd.DataFrame([features])
        scaled = scaler.transform(df)
        scaled = np.expand_dims(scaled, axis=2)

        payload = {"features": scaled[0].tolist()}
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp}, Class {result['predicted_class']}, Confidence {result['confidence']:.4f}"
            print(f"✅ {log_line}")
            with open(LOG_PATH, "a") as log_file:
                log_file.write(log_line + "\n")
        except Exception as e:
            print(f"❌ Error: {e}")
