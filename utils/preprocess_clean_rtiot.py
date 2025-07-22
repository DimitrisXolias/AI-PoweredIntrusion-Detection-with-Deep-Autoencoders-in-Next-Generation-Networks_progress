
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# === Paths ===
input_path = '/home/xold/ids-project/dataset/RT_IOT2022.CSV'
output_dir = '/home/xold/ids-project/data'
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(input_path)

# === Basic Cleaning ===
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# === Drop identifier columns ===
drop_cols = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ✅ Rename "Attack type" to "Label" before encoding

# === Encode categorical columns (including Label) ===
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# ✅ Verify "Label" exists
if 'Label' not in df.columns:
    raise Exception("❌ 'Label' column not found after encoding.")

# === Save cleaned dataset ===
cleaned_path = os.path.join(output_dir, 'cleaned.csv')
df.to_csv(cleaned_path, index=False)
print(f"[✓] Cleaned dataset saved to {cleaned_path}")

# === Split features and labels ===
X = df.drop('Label', axis=1)
y = df['Label']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Save splits ===
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False, header=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False, header=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=False)

print("[✓] Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")


