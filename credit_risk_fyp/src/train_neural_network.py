"""
Neural Network Training Script
Trains a feedforward neural network for credit risk assessment using SMOTE-balanced data.
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)

from src.data_pipeline import load_processed_data
from src.config import MODELS_DIR, FIGURES_DIR, RESULTS_DIR
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, set_seed

# Setup
setup_logging()
set_seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("NEURAL NETWORK MODEL TRAINING")
print("=" * 80)

# 1. Load Data
print("\n1. Loading processed data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# 2. Additional Scaling for Neural Network
print("\n2. Applying StandardScaler for Neural Network...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate class weight
pos_samples = y_train.sum()
neg_samples = len(y_train) - pos_samples
class_weight_ratio = neg_samples / pos_samples
class_weight = {0: 1.0, 1: class_weight_ratio}

print(f"Class weight ratio: {class_weight_ratio:.2f}")

# 3. Build Neural Network Model
print("\n3. Building Neural Network...")
print("Architecture: Input -> Dense(256) -> Dropout -> Dense(128) -> Dropout -> Dense(64) -> Dropout -> Output")

nn_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
], name='credit_risk_nn')

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

nn_model.summary()

# 4. Train Model
print("\n4. Training Neural Network...")
early_stop = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=2048,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 5. Make Predictions
print("\n5. Making predictions...")
y_val_proba = nn_model.predict(X_val_scaled, verbose=0).flatten()
y_test_proba = nn_model.predict(X_test_scaled, verbose=0).flatten()

# 6. Find Optimal Threshold
print("\n6. Finding optimal threshold...")
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)

# Optimize by sampling thresholds if there are too many
if len(thresholds) > 100:
    indices = np.linspace(0, len(thresholds)-1, 100, dtype=int)
    thresholds_sample = thresholds[indices]
else:
    thresholds_sample = thresholds

print(f"Evaluating {len(thresholds_sample)} threshold values...")
f1_scores = []
for i, threshold in enumerate(thresholds_sample):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(thresholds_sample)}")
    y_val_pred = (y_val_proba >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    f1_scores.append(f1)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_sample[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")

# 7. Evaluate on Test Set
print("\n7. Evaluating on test set...")
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

test_metrics = {
    'roc_auc': roc_auc_score(y_test, y_test_proba),
    'precision': precision_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

test_metrics['true_positives'] = int(tp)
test_metrics['false_positives'] = int(fp)
test_metrics['true_negatives'] = int(tn)
test_metrics['false_negatives'] = int(fn)
test_metrics['false_positive_rate'] = fp / (fp + tn)
test_metrics['true_positive_rate'] = tp / (tp + fn)
test_metrics['specificity'] = tn / (tn + fp)
test_metrics['optimal_threshold'] = float(optimal_threshold)

print("\n" + "=" * 80)
print("TEST SET RESULTS")
print("=" * 80)
print(f"AUC-ROC:          {test_metrics['roc_auc']:.4f}")
print(f"Precision:        {test_metrics['precision']:.4f}")
print(f"Recall (TPR):     {test_metrics['recall']:.4f}")
print(f"F1-Score:         {test_metrics['f1_score']:.4f}")
print(f"FPR:              {test_metrics['false_positive_rate']:.4f}")
print(f"Specificity:      {test_metrics['specificity']:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# 8. Save Results
print("\n8. Saving results...")

# Save model
model_path = MODELS_DIR / 'neural_network_model.keras'
nn_model.save(model_path)
print(f"Model saved to: {model_path}")

# Save scaler
scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_path}")

# Save metrics
metrics_path = RESULTS_DIR / 'neural_network_metrics.pkl'
with open(metrics_path, 'wb') as f:
    pickle.dump({
        'test_metrics': test_metrics,
        'optimal_threshold': optimal_threshold,
        'history': history.history
    }, f)
print(f"Metrics saved to: {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_proba': y_test_proba,
    'y_pred': y_test_pred
})
predictions_path = RESULTS_DIR / 'neural_network_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)