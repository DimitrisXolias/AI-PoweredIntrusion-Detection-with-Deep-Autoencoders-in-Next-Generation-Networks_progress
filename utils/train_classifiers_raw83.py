
# train_classifiers_raw83.py

import os
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore

# === Paths ===
data_dir = 'data'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

X_train = pd.read_csv('/home/xold/ids-project/data/X_train.csv', header=None).values
X_test = pd.read_csv('/home/xold/ids-project/data/X_test.csv', header=None).values
y_train = pd.read_csv('/home/xold/ids-project/data/y_train.csv', header=None).values
y_test = pd.read_csv('/home/xold/ids-project/data/y_test.csv', header=None).values

# === Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, f'{model_dir}/scaler_raw83.pkl')

# === PCA (SVM only) ===
pca = PCA(n_components=30, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
joblib.dump(pca, f'{model_dir}/pca_svm.pkl')

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
        MaxPooling1D(2), Dropout(0.5),
        Conv1D(128, 3, padding='same'), BatchNormalization(), LeakyReLU(),
        MaxPooling1D(2), Dropout(0.5),
        Flatten(),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)), BatchNormalization(), LeakyReLU(), Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model(X_train_cnn.shape[1:], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint(f'{model_dir}/1dcnn_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
]

history = model.fit(X_train_cnn, y_train_cat,
                    validation_data=(X_test_cnn, y_test_cat),
                    epochs=200,
                    batch_size=256,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=2)

cnn_preds = np.argmax(model.predict(X_test_cnn), axis=1)
cnn_acc = accuracy_score(y_test_enc, cnn_preds)

# === Save classification report ===
with open(f'{data_dir}/cnn_classification_report_raw83.txt', 'w') as f:
    f.write(classification_report(y_test_enc, cnn_preds))

# === CNN Confusion Matrix ===
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test_enc, cnn_preds)
ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
plt.title('CNN Confusion Matrix')
plt.savefig(f'{data_dir}/cnn_confusion_matrix_raw83.png')

# === CNN ROC + PR Curves ===
y_score = model.predict(X_test_cnn)
fpr, tpr, _ = roc_curve(y_test_cat.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.title('CNN ROC Curve')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_roc_curve_raw83.png')

precision, recall, _ = precision_recall_curve(y_test_cat.ravel(), y_score.ravel())
plt.figure()
plt.plot(recall, precision, color='purple')
plt.title('CNN Precision-Recall Curve')
plt.grid()
plt.tight_layout()
plt.savefig(f'{data_dir}/cnn_pr_curve_raw83.png')

# === SVM ===
svm = SVC(C=0.5, kernel='rbf', gamma='scale', probability=True)
svm.fit(X_train_pca, y_train_enc)
svm_preds = svm.predict(X_test_pca)
svm_probs = svm.predict_proba(X_test_pca)
svm_acc = accuracy_score(y_test_enc, svm_preds)
svm_f1 = f1_score(y_test_enc, svm_preds, average='macro')
svm_precision = precision_score(y_test_enc, svm_preds, average='macro')
svm_recall = recall_score(y_test_enc, svm_preds, average='macro')
svm_auc = roc_auc_score(y_test_cat, svm_probs, multi_class='ovr')
pickle.dump(svm, open(f'{model_dir}/svm_model.pkl', 'wb'))

svm_fpr, svm_tpr, _ = roc_curve(y_test_cat.ravel(), svm_probs.ravel())
svm_precision_curve, svm_recall_curve, _ = precision_recall_curve(y_test_cat.ravel(), svm_probs.ravel())

plt.figure()
plt.plot(svm_fpr, svm_tpr, label=f'SVM ROC (AUC = {svm_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.title('SVM ROC Curve')
plt.grid()
plt.legend()
plt.savefig(f'{data_dir}/svm_roc_curve_raw83.png')

plt.figure()
plt.plot(svm_recall_curve, svm_precision_curve, color='green')
plt.title('SVM Precision-Recall Curve')
plt.grid()
plt.savefig(f'{data_dir}/svm_pr_curve_raw83.png')

# === Random Forest ===
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_leaf=8,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train_enc)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)
rf_acc = accuracy_score(y_test_enc, rf_preds)
rf_f1 = f1_score(y_test_enc, rf_preds, average='macro')
rf_precision = precision_score(y_test_enc, rf_preds, average='macro')
rf_recall = recall_score(y_test_enc, rf_preds, average='macro')
rf_auc = roc_auc_score(y_test_cat, rf_probs, multi_class='ovr')
pickle.dump(rf, open(f'{model_dir}/rf_model.pkl', 'wb'))

rf_fpr, rf_tpr, _ = roc_curve(y_test_cat.ravel(), rf_probs.ravel())
rf_precision_curve, rf_recall_curve, _ = precision_recall_curve(y_test_cat.ravel(), rf_probs.ravel())

plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f'RF ROC (AUC = {rf_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.title('RF ROC Curve')
plt.grid()
plt.legend()
plt.savefig(f'{data_dir}/rf_roc_curve_raw83.png')

plt.figure()
plt.plot(rf_recall_curve, rf_precision_curve, color='orange')
plt.title('RF Precision-Recall Curve')
plt.grid()
plt.savefig(f'{data_dir}/rf_pr_curve_raw83.png')

# === Accuracy Comparison ===
plt.figure()
plt.bar(['CNN', 'SVM', 'RF'], [cnn_acc, svm_acc, rf_acc], color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{data_dir}/accuracy_comparison_raw83.png')

# === Confusion Matrices (SVM & RF) ===
for name, y_pred in zip(['svm', 'rf'], [svm_preds, rf_preds]):
    cm = confusion_matrix(y_test_enc, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
    plt.title(f'{name.upper()} Confusion Matrix')
    plt.savefig(f'{data_dir}/{name}_confusion_matrix_raw83.png')
    plt.close()

# === Save Predictions ===
np.savetxt(f'{data_dir}/y_pred_cnn_raw83.csv', cnn_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_svm_raw83.csv', svm_preds, fmt='%d', delimiter=',')
np.savetxt(f'{data_dir}/y_pred_rf_raw83.csv', rf_preds, fmt='%d', delimiter=',')

print("\n\n=== TRAINED MODELS SAVED ===")
print("1D-CNN: models/1dcnn_model.h5")
print("SVM:    models/svm_model.pkl")
print("RF:     models/rf_model.pkl")

print("\n=== EVALUATION METRICS ===")
print(f"CNN Accuracy: {cnn_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, AUC: {svm_auc:.4f}")
print(f"RF  Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}, Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, AUC: {rf_auc:.4f}")
