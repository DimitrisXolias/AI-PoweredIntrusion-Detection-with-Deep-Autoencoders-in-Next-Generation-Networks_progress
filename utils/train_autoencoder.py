
# train_autoencoder.py

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from keras.initializers import HeNormal
from keras.optimizers import Adam
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# === Config paths ===
data_path = "/home/xold/ids-project/data/cleaned.csv"
model_dir = "/home/xold/ids-project/models"
os.makedirs(model_dir, exist_ok=True)

# === Force CPU only ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === Load dataset ===
df = pd.read_csv(data_path)
X = df.iloc[:, :-1].values  # Drop label column

# === Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# === Train/Test split ===
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# === Add Gaussian noise for Denoising AE ===
X_train_noisy = X_train + np.random.normal(0, 0.02, X_train.shape)
X_test_noisy = X_test + np.random.normal(0, 0.02, X_test.shape)

# === Autoencoder architecture ===
input_dim = X_train.shape[1]  # This is 83
encoding_dim = 16
he_init = HeNormal()

input_layer = Input(shape=(input_dim,))
x = Dense(128, kernel_initializer=he_init)(input_layer)
x = ReLU()(x)
x = BatchNormalization()(x)

x = Dense(64, kernel_initializer=he_init)(x)
x = ReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32, kernel_initializer=he_init)(x)
x = ReLU()(x)

encoded = Dense(encoding_dim, activation='relu')(x)

x = Dense(32, kernel_initializer=he_init)(encoded)
x = ReLU()(x)

x = Dense(64, kernel_initializer=he_init)(x)
x = ReLU()(x)
x = Dropout(0.2)(x)

x = Dense(128, kernel_initializer=he_init)(x)
x = ReLU()(x)

decoded = Dense(input_dim, activation='linear')(x)

# === Compile and train model ===
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber())

callbacks = [
    EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1),
    ModelCheckpoint(filepath=os.path.join(model_dir, 'autoencoder_model.h5'), save_best_only=True)
]

history = autoencoder.fit(
    X_train_noisy, X_train,
    validation_data=(X_test_noisy, X_test),
    epochs=250,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# === Save encoded features ===
encoder = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

np.savetxt('/home/xold/ids-project/data/X_train_encoded.csv', X_train_encoded, delimiter=',')
np.savetxt('/home/xold/ids-project/data/X_test_encoded.csv', X_test_encoded, delimiter=',')

# === Plot training loss ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Huber Loss')
plt.title('Autoencoder Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/xold/ids-project/data/autoencoder_loss_plot.png')

print("[✓] Autoencoder trained.")
print("[✓] Encoded features saved.")
print("[✓] Loss plot generated.")

