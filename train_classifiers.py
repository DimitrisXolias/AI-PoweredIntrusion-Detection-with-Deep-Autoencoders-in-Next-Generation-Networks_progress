
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# === Paths ===
data_dir = "../data"
X_train_path = os.path.join(data_dir, "X_train_encoded.csv")
X_test_path = os.path.join(data_dir, "X_test_encoded.csv")
y_train_path = os.path.join(data_dir, "y_train.csv")
y_test_path = os.path.join(data_dir, "y_test.csv")
plot_dir = "../data"

# === Load Data ===
X_train = pd.read_csv(X_train_path, header=None).values
X_test = pd.read_csv(X_test_path, header=None).values
y_train = pd.read_csv(y_train_path, header=None).values.ravel()
y_test = pd.read_csv(y_test_path, header=None).values.ravel()

# === Normalize Features (again, just in case) ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 1D-CNN Classifier ===
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

cnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # binary classification
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("../data/cnn_model.keras", monitor='val_loss', save_best_only=True)

history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# === Save CNN Loss Plot ===
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('1D-CNN Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_dir, "cnn_loss_plot.png"))

# === SVM Classifier ===
print("\n[ SVM Training ]")
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_acc:.4f}")

# === Random Forest Classifier ===
print("\n[ RF Training ]")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# === Save Accuracy Bar Plot ===
models = ['1D-CNN', 'SVM', 'Random Forest']
accuracies = [
    history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0,
    svm_acc,
    rf_acc
]

plt.figure()
plt.bar(models, accuracies, color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
plt.savefig(os.path.join(plot_dir, "model_comparison_plot.png"))

print("\n✓ All models trained and plots saved to ../data/")
