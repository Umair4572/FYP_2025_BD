"""
Logistic Regression Training Script
Trains a logistic regression model for credit risk assessment using SMOTE-balanced data.
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

from src.data_pipeline import load_processed_data
from src.config import MODELS_DIR, RESULTS_DIR
from src.utils import setup_logging, set_seed

# Setup
setup_logging()
set_seed(42)

print("=" * 80)
print("LOGISTIC REGRESSION MODEL TRAINING")
print("=" * 80)

# 1. Load Data
print("\n1. Loading processed data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# 2. Train Logistic Regression
print("\n2. Training Logistic Regression...")
print("Parameters: C=1.0, max_iter=1000, solver='lbfgs'")

lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train, y_train)
print("Training complete!")

# 3. Make Predictions
print("\n3. Making predictions...")
y_val_proba = lr_model.predict_proba(X_val)[:, 1]
y_test_proba = lr_model.predict_proba(X_test)[:, 1]

# 4. Find Optimal Threshold
print("\n4. Finding optimal threshold...")
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

# 5. Evaluate on Test Set
print("\n5. Evaluating on test set...")
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

# 6. Save Results
print("\n6. Saving results...")

# Save model
model_path = MODELS_DIR / 'logistic_regression_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(lr_model, f)
print(f"Model saved to: {model_path}")

# Save metrics
metrics_path = RESULTS_DIR / 'logistic_regression_metrics.pkl'
with open(metrics_path, 'wb') as f:
    pickle.dump({
        'test_metrics': test_metrics,
        'optimal_threshold': optimal_threshold
    }, f)
print(f"Metrics saved to: {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_proba': y_test_proba,
    'y_pred': y_test_pred
})
predictions_path = RESULTS_DIR / 'logistic_regression_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
