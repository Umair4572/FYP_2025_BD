"""
Ensemble Model Utilities
Helper functions for stacking and weighted averaging ensembles
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'


def load_base_model_predictions() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all base model predictions from saved pickle files.

    Returns:
        Dictionary with structure:
        {
            'logistic_regression': {'train': probs, 'val': probs, 'test': probs, 'y_test': labels},
            'random_forest': {...},
            'xgboost': {...},
            'neural_network': {...}
        }
    """
    print("\n" + "="*80)
    print("LOADING BASE MODEL PREDICTIONS")
    print("="*80)

    base_models = {
        'logistic_regression': 'logistic_regression_metrics.pkl',
        'random_forest': 'random_forest_metrics.pkl',
        'xgboost': 'xgboost_metrics.pkl',
        'neural_network': 'neural_network_metrics.pkl'
    }

    predictions = {}

    for model_name, metrics_file in base_models.items():
        metrics_path = RESULTS_DIR / metrics_file
        predictions_csv = RESULTS_DIR / f"{model_name}_predictions.csv"

        if not metrics_path.exists():
            print(f"WARNING: {metrics_file} not found. Skipping {model_name}...")
            continue

        # Load metrics (contains test predictions and labels)
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        # Load predictions CSV for train/val if available
        if predictions_csv.exists():
            df = pd.read_csv(predictions_csv)
            predictions[model_name] = {
                'train_probs': df[df['split'] == 'train']['probability'].values if 'train' in df['split'].values else None,
                'val_probs': df[df['split'] == 'validation']['probability'].values if 'validation' in df['split'].values else None,
                'test_probs': df[df['split'] == 'test']['probability'].values,
                'y_test': df[df['split'] == 'test']['true_label'].values,
            }
        else:
            # Fallback: only test predictions available from metrics
            predictions[model_name] = {
                'train_probs': None,
                'val_probs': None,
                'test_probs': metrics.get('test_probs', None),
                'y_test': metrics.get('y_test', None),
            }

        print(f"✓ Loaded {model_name}")

    print(f"\nTotal models loaded: {len(predictions)}")
    return predictions


def load_base_models() -> Dict[str, any]:
    """
    Load all trained base models.

    Returns:
        Dictionary of loaded model objects
    """
    print("\n" + "="*80)
    print("LOADING BASE MODELS")
    print("="*80)

    models = {}

    # Logistic Regression
    lr_path = MODELS_DIR / 'logistic_regression_model.pkl'
    if lr_path.exists():
        with open(lr_path, 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
        print("✓ Loaded Logistic Regression")

    # Random Forest
    rf_path = MODELS_DIR / 'random_forest_model.pkl'
    if rf_path.exists():
        with open(rf_path, 'rb') as f:
            models['random_forest'] = pickle.load(f)
        print("✓ Loaded Random Forest")

    # XGBoost
    xgb_path = MODELS_DIR / 'xgboost_model.pkl'
    if xgb_path.exists():
        try:
            with open(xgb_path, 'rb') as f:
                models['xgboost'] = pickle.load(f)
            print("✓ Loaded XGBoost")
        except Exception as e:
            print(f"⚠ Could not load XGBoost model: {str(e)[:60]}...")
            print("  (Skipping XGBoost - please retrain with new data)")

    # Neural Network
    nn_path = MODELS_DIR / 'neural_network_model.keras'
    if nn_path.exists():
        try:
            import tensorflow as tf
            models['neural_network'] = tf.keras.models.load_model(nn_path)
            print("✓ Loaded Neural Network")
        except ImportError:
            print("⚠ TensorFlow not available in this environment, skipping Neural Network model")
            print("  (Ensemble will work with remaining models)")

    print(f"\nTotal models loaded: {len(models)}")
    return models


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix for FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auc_roc': auc_roc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'specificity': specificity,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'tpr_fpr', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_metrics)
    """
    print(f"\nFinding optimal threshold (optimizing {metric})...")

    # Sample 100 thresholds between min and max probability
    thresholds = np.linspace(y_pred_proba.min(), y_pred_proba.max(), 100)

    best_score = -np.inf
    best_threshold = 0.5
    best_metrics = {}

    for i, threshold in enumerate(thresholds):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(thresholds)}")

        metrics = calculate_metrics(y_true, y_pred_proba, threshold)

        # Calculate optimization score
        if metric == 'f1':
            score = metrics['f1_score']
        elif metric == 'tpr_fpr':
            score = metrics['recall'] - metrics['fpr']  # TPR - FPR
        elif metric == 'precision':
            score = metrics['precision']
        elif metric == 'recall':
            score = metrics['recall']
        else:
            score = metrics['f1_score']

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    print(f"Optimal threshold: {best_threshold:.4f}")
    return best_threshold, best_metrics


def save_ensemble_results(model_name: str, test_metrics: Dict, optimal_threshold: float,
                         y_test: np.ndarray, test_probs: np.ndarray):
    """
    Save ensemble model results to files.

    Args:
        model_name: Name of the ensemble model
        test_metrics: Dictionary of test metrics
        optimal_threshold: Optimal classification threshold
        y_test: True test labels
        test_probs: Predicted test probabilities
    """
    print(f"\nSaving {model_name} results...")

    # Save metrics
    metrics_to_save = {
        'test_metrics': test_metrics,
        'optimal_threshold': optimal_threshold,
        'y_test': y_test,
        'test_probs': test_probs
    }

    metrics_path = RESULTS_DIR / f'{model_name}_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_to_save, f)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Save predictions CSV
    test_preds = (test_probs >= optimal_threshold).astype(int)

    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': test_preds,
        'probability': test_probs,
        'split': 'test'
    })

    predictions_path = RESULTS_DIR / f'{model_name}_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")


def print_metrics(metrics: Dict[str, float], title: str = "METRICS"):
    """Pretty print metrics."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"AUC-ROC:          {metrics['auc_roc']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall (TPR):     {metrics['recall']:.4f}")
    print(f"F1-Score:         {metrics['f1_score']:.4f}")
    print(f"FPR:              {metrics['fpr']:.4f}")
    print(f"Specificity:      {metrics['specificity']:.4f}")