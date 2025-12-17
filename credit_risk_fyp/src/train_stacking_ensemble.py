"""
Stacking Ensemble Model Training
Combines Logistic Regression, Random Forest, XGBoost, and Neural Network
Uses Logistic Regression as meta-learner with 5-fold cross-validation
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def generate_meta_features(base_models: dict, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, X_test: np.ndarray,
                          n_folds: int = 5) -> tuple:
    """
    Generate meta-features using out-of-fold predictions.

    Args:
        base_models: Dictionary of trained base models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        X_test: Test features
        n_folds: Number of cross-validation folds

    Returns:
        Tuple of (train_meta_features, val_meta_features, test_meta_features)
    """
    print("\n" + "="*80)
    print("GENERATING META-FEATURES WITH 5-FOLD CROSS-VALIDATION")
    print("="*80)

    n_models = len(base_models)
    n_train = X_train.shape[0]

    # Initialize meta-features arrays
    train_meta = np.zeros((n_train, n_models))
    val_meta = np.zeros((X_val.shape[0], n_models))
    test_meta = np.zeros((X_test.shape[0], n_models))

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    model_names = list(base_models.keys())

    print(f"\nBase models: {', '.join(model_names)}")
    print(f"Folds: {n_folds}")
    print(f"Training samples: {n_train:,}")

    # For each base model
    for model_idx, (model_name, model) in enumerate(base_models.items()):
        print(f"\n[{model_idx + 1}/{n_models}] Processing {model_name}...")

        # Out-of-fold predictions for training set
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]

            # Clone and train model on this fold
            if model_name == 'logistic_regression':
                fold_model = LogisticRegression(
                    C=1.0, max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE
                )
                fold_model.fit(X_fold_train, y_fold_train)
                fold_pred = fold_model.predict_proba(X_fold_val)
                train_meta[val_idx, model_idx] = fold_pred[:, 1] if fold_pred.ndim == 2 else fold_pred

            elif model_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                fold_model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1
                )
                fold_model.fit(X_fold_train, y_fold_train)
                fold_pred = fold_model.predict_proba(X_fold_val)
                train_meta[val_idx, model_idx] = fold_pred[:, 1] if fold_pred.ndim == 2 else fold_pred

            elif model_name == 'xgboost':
                import xgboost as xgb
                scale_pos_weight = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()
                fold_model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
                    eval_metric='logloss'
                )
                fold_model.fit(X_fold_train, y_fold_train)
                fold_pred = fold_model.predict_proba(X_fold_val)
                train_meta[val_idx, model_idx] = fold_pred[:, 1] if fold_pred.ndim == 2 else fold_pred

            elif model_name == 'neural_network':
                import tensorflow as tf
                from sklearn.preprocessing import StandardScaler as Scaler

                # Scale features for NN
                scaler = Scaler()
                X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                X_fold_val_scaled = scaler.transform(X_fold_val)

                # Use smaller NN for speed
                fold_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_dim=X_fold_train.shape[1]),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])

                fold_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                fold_model.fit(
                    X_fold_train_scaled, y_fold_train,
                    epochs=10, batch_size=2048, verbose=0
                )

                train_meta[val_idx, model_idx] = fold_model.predict(X_fold_val_scaled, verbose=0).ravel()

            print("Done")

        # Full model predictions for validation and test sets
        print(f"  Generating predictions for validation and test sets...")

        if model_name == 'neural_network':
            # Load scaler for NN
            scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                val_meta[:, model_idx] = model.predict(X_val_scaled, verbose=0).ravel()
                test_meta[:, model_idx] = model.predict(X_test_scaled, verbose=0).ravel()
            else:
                print("  WARNING: Scaler not found, using unscaled features")
                val_meta[:, model_idx] = model.predict(X_val, verbose=0).ravel()
                test_meta[:, model_idx] = model.predict(X_test, verbose=0).ravel()
        else:
            # Get predictions
            val_pred = model.predict_proba(X_val)
            test_pred = model.predict_proba(X_test)

            # Handle both 1D and 2D arrays
            if val_pred.ndim == 1:
                val_meta[:, model_idx] = val_pred
                test_meta[:, model_idx] = test_pred
            else:
                val_meta[:, model_idx] = val_pred[:, 1]
                test_meta[:, model_idx] = test_pred[:, 1]

        print(f"  ✓ {model_name} meta-features complete")

    print("\n" + "="*80)
    print("META-FEATURES GENERATION COMPLETE")
    print("="*80)
    print(f"Train meta-features shape: {train_meta.shape}")
    print(f"Validation meta-features shape: {val_meta.shape}")
    print(f"Test meta-features shape: {test_meta.shape}")

    return train_meta, val_meta, test_meta


def train_meta_learner(train_meta: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train meta-learner (Logistic Regression).

    Args:
        train_meta: Meta-features from base models
        y_train: Training labels

    Returns:
        Trained meta-learner
    """
    print("\n" + "="*80)
    print("TRAINING META-LEARNER (LOGISTIC REGRESSION)")
    print("="*80)

    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=RANDOM_STATE
    )

    meta_learner.fit(train_meta, y_train)

    print("✓ Meta-learner training complete")
    print(f"\nMeta-learner coefficients:")
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
    for name, coef in zip(model_names, meta_learner.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    return meta_learner


def main():
    """Main training pipeline for stacking ensemble."""
    print("\n" + "="*80)
    print("STACKING ENSEMBLE MODEL TRAINING")
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
        print("ERROR: Need at least 2 base models for stacking!")
        return

    # 3. Generate meta-features
    print("\n3. Generating meta-features...")
    train_meta, val_meta, test_meta = generate_meta_features(
        base_models, X_train, y_train, X_val, X_test, n_folds=5
    )

    # 4. Train meta-learner
    print("\n4. Training meta-learner...")
    meta_learner = train_meta_learner(train_meta, y_train)

    # 5. Make predictions
    print("\n5. Making predictions...")
    val_probs = meta_learner.predict_proba(val_meta)[:, 1]
    test_probs = meta_learner.predict_proba(test_meta)[:, 1]

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

    # Save meta-learner
    meta_learner_path = MODELS_DIR / 'stacking_ensemble_meta.pkl'
    with open(meta_learner_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    print(f"✓ Meta-learner saved to: {meta_learner_path}")

    # Save results
    save_ensemble_results(
        'stacking_ensemble',
        test_metrics,
        optimal_threshold,
        y_test,
        test_probs
    )

    print("\n" + "="*80)
    print("STACKING ENSEMBLE TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()