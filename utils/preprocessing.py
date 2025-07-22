# utils/preprocessing.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(input_csv, label_column):
    df = pd.read_csv(input_csv)
    y = df[label_column].values
    X = df.drop(columns=[label_column]).values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaled data
    pd.DataFrame(X_train_scaled).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_val_scaled).to_csv("data/X_val.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_val).to_csv("data/y_val.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)
    

    joblib.dump(scaler, "models/scaler.pkl")

