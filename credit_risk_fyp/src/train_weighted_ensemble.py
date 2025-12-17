"""
Weighted Averaging Ensemble Model Training
Optimizes linear combination weights for base models using scipy
Simpler alternative to stacking with interpretable weights
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_utils import (
    load_base_models, calculate_metrics, find_optimal_threshold,
    save_ensemble_results, print_metrics, MODELS_DIR, RESULTS_DIR
)
from data_pipeline import load_processed_data

# Random state for reproducibility
RANDOM_STATE = 42


def get_base_predictions(base_models: dict, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Get predictions from all base models.

    Args:
        base_models: Dictionary of trained base models
        X_val: Validation features
        X_test: Test features

    Returns:
        Tuple of (val_predictions_matrix, test_predictions_matrix)
        Each matrix has shape (n_samples, n_models)
    """
    print("\n" + "="*80)
    print("GETTING BASE MODEL PREDICTIONS")
    print("="*80)

    n_models = len(base_models)
    val_preds = np.zeros((X_val.shape[0], n_models))
    test_preds = np.zeros((X_test.shape[0], n_models))

    model_names = list(base_models.keys())
    print(f"Base models: {', '.join(model_names)}")

    for idx, (model_name, model) in enumerate(base_models.items()):
        print(f"\n[{idx + 1}/{n_models}] {model_name}...")

        if model_name == 'neural_network':
            # Load scaler for NN
            scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                val_preds[:, idx] = model.predict(X_val_scaled, verbose=0).ravel()
                test_preds[:, idx] = model.predict(X_test_scaled, verbose=0).ravel()
            else:
                print("  WARNING: Scaler not found, using unscaled features")
                val_preds[:, idx] = model.predict(X_val, verbose=0).ravel()
                test_preds[:, idx] = model.predict(X_test, verbose=0).ravel()
        else:
            # Get predictions
            val_pred = model.predict_proba(X_val)
            test_pred = model.predict_proba(X_test)

            # Handle both 1D and 2D arrays
            if val_pred.ndim == 1:
                val_preds[:, idx] = val_pred
                test_preds[:, idx] = test_pred
            else:
                val_preds[:, idx] = val_pred[:, 1]
                test_preds[:, idx] = test_pred[:, 1]

        print(f"  ✓ Predictions shape: val={val_preds[:, idx].shape}, test={test_preds[:, idx].shape}")

    print("\n" + "="*80)
    print("BASE PREDICTIONS COMPLETE")
    print("="*80)

    return val_preds, test_preds


def optimize_weights(val_preds: np.ndarray, y_val: np.ndarray, metric: str = 'auc_roc') -> np.ndarray:
    """
    Optimize ensemble weights using scipy.

    Args:
        val_preds: Validation predictions matrix (n_samples, n_models)
        y_val: Validation labels
        metric: Metric to optimize ('auc_roc', 'f1', 'tpr_fpr')

    Returns:
        Optimized weights array
    """
    print("\n" + "="*80)
    print(f"OPTIMIZING WEIGHTS (metric: {metric.upper()})")
    print("="*80)

    n_models = val_preds.shape[1]

    # Objective function (negative because we minimize)
    def objective(weights):
        # Weighted average predictions
        weighted_pred = np.dot(val_preds, weights)

        # Calculate metric
        metrics = calculate_metrics(y_val, weighted_pred, threshold=0.5)

        if metric == 'auc_roc':
            score = metrics['auc_roc']
        elif metric == 'f1':
            score = metrics['f1_score']
        elif metric == 'tpr_fpr':
            score = metrics['recall'] - metrics['fpr']  # TPR - FPR
        else:
            score = metrics['auc_roc']

        return -score  # Negative for minimization

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    # Bounds: each weight between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Initial guess: equal weights
    initial_weights = np.ones(n_models) / n_models

    print(f"Initial weights (equal): {initial_weights}")
    print("Running optimization...")

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': False}
    )

    if not result.success:
        print(f"WARNING: Optimization did not converge: {result.message}")
        print("Using equal weights as fallback")
        return initial_weights

    optimized_weights = result.x

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
    print("\nOptimized weights:")
    for name, weight in zip(model_names, optimized_weights):
        print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")

    print(f"\nSum of weights: {optimized_weights.sum():.6f}")
    print(f"Optimization score: {-result.fun:.4f}")

    return optimized_weights


def main():
    """Main training pipeline for weighted averaging ensemble."""
    print("\n" + "="*80)
    print("WEIGHTED AVERAGING ENSEMBLE MODEL TRAINING")
    print("="*80)

    # 1. Load processed data
    print("\n1. Loading processed data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()

    # Convert to numpy arrays if needed
    X_train = X_train.values if hasattr(X_train, 'values') else X_train
    X_val = X_val.values if hasattr(X_val, 'values') else X_val
    X_test = X_test.values if hasattr(X_test, 'values') else X_test
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_val = y_val.values if hasattr(y_val, 'values') else y_val
    y_test = y_test.values if hasattr(y_test, 'values') else y_test

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # 2. Load base models
    print("\n2. Loading base models...")
    base_models = load_base_models()

    if len(base_models) < 2:
        print("ERROR: Need at least 2 base models for weighted ensemble!")
        return

    # 3. Get base model predictions
    print("\n3. Getting base model predictions...")
    val_preds, test_preds = get_base_predictions(base_models, X_val, X_test)

    # 4. Optimize weights
    print("\n4. Optimizing ensemble weights...")
    optimized_weights = optimize_weights(val_preds, y_val, metric='auc_roc')

    # 5. Make weighted predictions
    print("\n5. Making weighted predictions...")
    val_probs = np.dot(val_preds, optimized_weights)
    test_probs = np.dot(test_preds, optimized_weights)

    # 6. Find optimal threshold
    print("\n6. Finding optimal threshold on validation set...")
    optimal_threshold, val_metrics = find_optimal_threshold(y_val, val_probs, metric='tpr_fpr')
    print_metrics(val_metrics, "VALIDATION SET RESULTS")

    # 7. Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_metrics = calculate_metrics(y_test, test_probs, optimal_threshold)
    test_metrics['optimal_threshold'] = optimal_threshold
    print_metrics(test_metrics, "TEST SET RESULTS")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # 8. Save results
    print("\n8. Saving results...")

    # Save optimized weights
    weights_dict = {
        'weights': optimized_weights,
        'model_names': list(base_models.keys())
    }
    weights_path = MODELS_DIR / 'weighted_ensemble_weights.pkl'
    with open(weights_path, 'wb') as f:
        pickle.dump(weights_dict, f)
    print(f"✓ Weights saved to: {weights_path}")

    # Save results
    save_ensemble_results(
        'weighted_ensemble',
        test_metrics,
        optimal_threshold,
        y_test,
        test_probs
    )

    print("\n" + "="*80)
    print("WEIGHTED ENSEMBLE TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()