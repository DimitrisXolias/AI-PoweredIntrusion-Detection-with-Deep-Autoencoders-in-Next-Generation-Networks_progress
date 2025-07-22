
import os
import subprocess
import time
import datetime
import pandas as pd
import numpy as np
import requests
import pickle
import joblib

# === PATHS ===
BASE_DIR = "/home/xold/ids-project"
DATA_DIR = os.path.join(BASE_DIR, "data")
PCAP_PATH = os.path.join(DATA_DIR, "capture.pcap")
CSV_PATH = os.path.join(DATA_DIR, "features.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_raw83.pkl")
LOG_PATH = os.path.join(DATA_DIR, "results.log")
FEATURE_SCRIPT = os.path.join(BASE_DIR, "scripts", "feature_extraction.py")
API_URL = "http://127.0.0.1:5000/predict"

# === STEP 1: Capture packets ===
def capture_traffic(interface="r1-eth2", duration=10):
    print(f"📡 Capturing traffic on {interface} for {duration} seconds...")
    try:
        subprocess.run(["sudo", "timeout", str(duration), "tcpdump", "-i", interface, "-w", PCAP_PATH])
        print("✅ Capture complete")
    except Exception as e:
        print(f"❌ Tcpdump failed: {e}")

# === STEP 2: Feature Extraction ===
def extract_features():
    if not os.path.exists(FEATURE_SCRIPT):
        print(f"❌ Feature extraction script not found: {FEATURE_SCRIPT}")
        return
    print("🔍 Extracting features from PCAP...")
    subprocess.run(["python3", FEATURE_SCRIPT])

# === STEP 3: Predict each flow ===
def predict_flows():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Features file not found: {CSV_PATH}")
        return
    try:
        df = pd.read_csv(CSV_PATH)
        scaler = joblib.load(open(SCALER_PATH, "rb"))
        X_scaled = scaler.transform(df)
        X_scaled = np.expand_dims(X_scaled, axis=2)

        for i, sample in enumerate(X_scaled):
            payload = {"features": sample.tolist()}
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"{timestamp}, Row {i}, Class {result['predicted_class']}, Confidence {result['confidence']:.4f}"
                print("✅", log_entry)
                with open(LOG_PATH, "a") as f:
                    f.write(log_entry + "\n")
            else:
                print(f"❌ API responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")

# === MAIN ===
if __name__ == "__main__":
    print("🚀 Starting IDS Inference Pipeline...\n")
    capture_traffic(interface="r1-eth2", duration=10)
    extract_features()
    predict_flows()
    print("\n🎯 Pipeline completed.")
