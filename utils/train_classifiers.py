
# train_classifiers_raw83.py
import os
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Paths ===
data_dir = 'data'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

X_train = pd.read_csv(f'{data_dir}/X_train.csv', header=None).values
X_test = pd.read_csv(f'{data_dir}/X_test.csv', header=None).values
y_train = pd.read_csv(f'{data_dir}/y_train.csv', header=None).values.ravel()
y_test = pd.read_csv(f'{data_dir}/y_test.csv', header=None).values.ravel()

# === Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, f'{model_dir}/scaler_raw83.pkl')

# === Label encoding ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
num_classes = len(np.unique(y_train_enc))

y_train_cat = to_categorical(y_train_enc, num_classes)
y_test_cat = to_categorical(y_test_enc, num_classes)

# === CNN input shape ===
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Class weights ===
classes = np.unique(y_train_enc)
weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
class_weight_dict = dict(zip(classes, weights))

# === CNN Model ===
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, padding='same'), BatchNormalization(), LeakyReLU(),
        MaxPooling1D(2), Dropout(0.4),
        Conv1D(128, 3, padding='same'), BatchNormalization(), LeakyReLU(),
        MaxPooling1D(2), Dropout(0.4),
        Flatten(),
        Dense(128), BatchNormalization(), LeakyReLU(), Dropout(0.5),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model(X_train_cnn.shape[1:], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
cnn_model_path = f'{model_dir}/1dcnn_model.h5'
callbacks = [
    EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint(cnn_model_path, save_best_only=True)
]

# === Fit CNN ===
history = model.fit(X_train_cnn, y_train_cat,
                    validation_data=(X_test_cnn, y_test_cat),
                    epochs=100,
                    batch_size=128,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=2)

# === CNN Eval ===
cnn_preds = np.argmax(model.predict(X_test_cnn), axis=1)
cnn_acc = accuracy_score(y_test_enc, cnn_preds)

# === Save accuracy curve ===
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('CNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_accuracy_curve_raw83.png')

# === SVM ===
svm = SVC(C=1.0, kernel='rbf', gamma='scale')
svm.fit(X_train, y_train_enc)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test_enc, svm_preds)
pickle.dump(svm, open(f'{model_dir}/svm_model.pkl', 'wb'))

# === RF ===
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train_enc)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test_enc, rf_preds)
pickle.dump(rf, open(f'{model_dir}/rf_model.pkl', 'wb'))

# === Accuracy bar plot ===
plt.figure()
plt.bar(['CNN', 'SVM', 'RF'], [cnn_acc, svm_acc, rf_acc], color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{data_dir}/accuracy_comparison_raw83.png')

# === Confusion Matrix ===
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test_enc, cnn_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', values_format='d')
plt.title('CNN Confusion Matrix')
plt.savefig(f'{data_dir}/cnn_confusion_matrix_raw83.png')

# === ROC Curve ===
y_score = model.predict(X_test_cnn)
fpr, tpr, _ = roc_curve(y_test_cat.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('CNN ROC Curve')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_roc_curve_raw83.png')

# === PR Curve ===
precision, recall, _ = precision_recall_curve(y_test_cat.ravel(), y_score.ravel())
plt.figure()
plt.plot(recall, precision, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('CNN Precision-Recall Curve')
plt.grid()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_pr_curve_raw83.png')

# === Save classification report
with open(f'{data_dir}/cnn_classification_report_raw83.txt', 'w') as f:
    f.write(classification_report(y_test_enc, cnn_preds))

# === Save predictions
np.savetxt(f'{data_dir}/y_pred_cnn_raw83.csv', cnn_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_svm_raw83.csv', svm_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_rf_raw83.csv', rf_preds, fmt='%d', delimiter=',')

print(f"[✓] Training complete. Models and metrics saved.")
# train_classifiers_raw83.py
import os
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Paths ===
data_dir = 'data'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

X_train = pd.read_csv(f'{data_dir}/X_train.csv', header=None).values
X_test = pd.read_csv(f'{data_dir}/X_test.csv', header=None).values
y_train = pd.read_csv(f'{data_dir}/y_train.csv', header=None).values.ravel()
y_test = pd.read_csv(f'{data_dir}/y_test.csv', header=None).values.ravel()

# === Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, f'{model_dir}/scaler_raw83.pkl')

# === Label encoding ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
num_classes = len(np.unique(y_train_enc))

y_train_cat = to_categorical(y_train_enc, num_classes)
y_test_cat = to_categorical(y_test_enc, num_classes)

# === CNN input shape ===
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Class weights ===
classes = np.unique(y_train_enc)
weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
class_weight_dict = dict(zip(classes, weights))

# === CNN Model ===
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, padding='same'), BatchNormalization(), LeakyReLU(),
        MaxPooling1D(2), Dropout(0.4),
        Conv1D(128, 3, padding='same'), BatchNormalization(), LeakyReLU(),
        MaxPooling1D(2), Dropout(0.4),
        Flatten(),
        Dense(128), BatchNormalization(), LeakyReLU(), Dropout(0.5),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model(X_train_cnn.shape[1:], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
cnn_model_path = f'{model_dir}/1dcnn_model.h5'
callbacks = [
    EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint(cnn_model_path, save_best_only=True)
]

# === Fit CNN ===
history = model.fit(X_train_cnn, y_train_cat,
                    validation_data=(X_test_cnn, y_test_cat),
                    epochs=100,
                    batch_size=128,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=2)

# === CNN Eval ===
cnn_preds = np.argmax(model.predict(X_test_cnn), axis=1)
cnn_acc = accuracy_score(y_test_enc, cnn_preds)

# === Save accuracy curve ===
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('CNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_accuracy_curve_raw83.png')

# === SVM ===
svm = SVC(C=1.0, kernel='rbf', gamma='scale')
svm.fit(X_train, y_train_enc)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test_enc, svm_preds)
pickle.dump(svm, open(f'{model_dir}/svm_model.pkl', 'wb'))

# === RF ===
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train_enc)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test_enc, rf_preds)
pickle.dump(rf, open(f'{model_dir}/rf_model.pkl', 'wb'))

# === Accuracy bar plot ===
plt.figure()
plt.bar(['CNN', 'SVM', 'RF'], [cnn_acc, svm_acc, rf_acc], color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{data_dir}/accuracy_comparison_raw83.png')

# === Confusion Matrix ===
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test_enc, cnn_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', values_format='d')
plt.title('CNN Confusion Matrix')
plt.savefig(f'{data_dir}/cnn_confusion_matrix_raw83.png')

# === ROC Curve ===
y_score = model.predict(X_test_cnn)
fpr, tpr, _ = roc_curve(y_test_cat.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('CNN ROC Curve')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_roc_curve_raw83.png')

# === PR Curve ===
precision, recall, _ = precision_recall_curve(y_test_cat.ravel(), y_score.ravel())
plt.figure()
plt.plot(recall, precision, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('CNN Precision-Recall Curve')
plt.grid()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_pr_curve_raw83.png')

# === Save classification report
with open(f'{data_dir}/cnn_classification_report_raw83.txt', 'w') as f:
    f.write(classification_report(y_test_enc, cnn_preds))

# === Save predictions
np.savetxt(f'{data_dir}/y_pred_cnn_raw83.csv', cnn_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_svm_raw83.csv', svm_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_rf_raw83.csv', rf_preds, fmt='%d', delimiter=',')

print(f"[✓] Training complete. Models and metrics saved.")
