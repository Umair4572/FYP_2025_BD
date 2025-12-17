"""
Expected Loss Calculator
Combines PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default)
Expected Loss = PD × LGD × EAD
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'


class ExpectedLossCalculator:
    """Calculator for Expected Loss using PD, LGD, and EAD models."""

    def __init__(self):
        """Initialize by loading all models."""
        self.pd_model = None
        self.lgd_rf_model = None
        self.lgd_xgb_model = None
        self.ead_rf_model = None
        self.ead_xgb_model = None
        self.lgd_features = None
        self.ead_features = None

    def load_models(self):
        """Load all trained models."""
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80)

        # Load PD model (Weighted Ensemble)
        try:
            pd_weights_path = MODELS_DIR / 'weighted_ensemble_weights.pkl'
            with open(pd_weights_path, 'rb') as f:
                pd_data = pickle.load(f)
            self.pd_weights = pd_data['weights']
            self.pd_model_names = pd_data['model_names']

            # Load base models for PD
            self.pd_models = {}
            models_loaded = []
            for name in self.pd_model_names:
                try:
                    if name == 'logistic_regression':
                        path = MODELS_DIR / 'logistic_regression_model.pkl'
                    elif name == 'random_forest':
                        path = MODELS_DIR / 'random_forest_model.pkl'
                    elif name == 'xgboost':
                        path = MODELS_DIR / 'xgboost_model.pkl'
                    elif name == 'neural_network':
                        path = MODELS_DIR / 'neural_network_model.keras'
                        import tensorflow as tf
                        self.pd_models[name] = tf.keras.models.load_model(path)
                        # Load scaler
                        scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
                        with open(scaler_path, 'rb') as f:
                            self.nn_scaler = pickle.load(f)
                        models_loaded.append(name)
                        continue

                    with open(path, 'rb') as f:
                        self.pd_models[name] = pickle.load(f)
                    models_loaded.append(name)
                except Exception as e:
                    print(f"  Warning: Could not load {name}: {str(e)[:60]}")
                    continue

            # Update weights to only include loaded models
            if len(models_loaded) < len(self.pd_model_names):
                loaded_indices = [i for i, name in enumerate(self.pd_model_names) if name in models_loaded]
                self.pd_weights = self.pd_weights[loaded_indices]
                self.pd_weights = self.pd_weights / self.pd_weights.sum()  # Renormalize
                self.pd_model_names = models_loaded

            print(f"[OK] PD Model (Weighted Ensemble) loaded with {len(models_loaded)} models")
        except Exception as e:
            print(f"[FAIL] PD Model loading failed: {str(e)}")

        # Load LGD models
        try:
            lgd_rf_path = MODELS_DIR / 'lgd_random_forest.pkl'
            lgd_xgb_path = MODELS_DIR / 'lgd_xgboost.pkl'

            with open(lgd_rf_path, 'rb') as f:
                self.lgd_rf_model = pickle.load(f)
            with open(lgd_xgb_path, 'rb') as f:
                self.lgd_xgb_model = pickle.load(f)

            # Load feature columns
            lgd_metrics_path = RESULTS_DIR / 'lgd_metrics.pkl'
            with open(lgd_metrics_path, 'rb') as f:
                lgd_data = pickle.load(f)
            self.lgd_features = lgd_data['feature_columns']

            print("[OK] LGD Models (RF + XGBoost) loaded")
        except Exception as e:
            print(f"[FAIL] LGD Models loading failed: {str(e)}")

        # Load EAD models
        try:
            ead_rf_path = MODELS_DIR / 'ead_random_forest.pkl'
            ead_xgb_path = MODELS_DIR / 'ead_xgboost.pkl'

            with open(ead_rf_path, 'rb') as f:
                self.ead_rf_model = pickle.load(f)
            with open(ead_xgb_path, 'rb') as f:
                self.ead_xgb_model = pickle.load(f)

            # Load feature columns
            ead_metrics_path = RESULTS_DIR / 'ead_metrics.pkl'
            with open(ead_metrics_path, 'rb') as f:
                ead_data = pickle.load(f)
            self.ead_features = ead_data['feature_columns']

            print("[OK] EAD Models (RF + XGBoost) loaded")
        except Exception as e:
            print(f"[FAIL] EAD Models loading failed: {str(e)}")

    def predict_pd(self, X):
        """Predict Probability of Default."""
        predictions = []

        for i, name in enumerate(self.pd_model_names):
            model = self.pd_models[name]

            if name == 'neural_network':
                X_scaled = self.nn_scaler.transform(X)
                pred = model.predict(X_scaled, verbose=0).ravel()
            else:
                pred = model.predict_proba(X)
                pred = pred[:, 1] if pred.ndim == 2 else pred

            predictions.append(pred)

        # Weighted average
        pd_probs = np.average(predictions, axis=0, weights=self.pd_weights)
        return pd_probs

    def predict_lgd(self, X):
        """Predict Loss Given Default (ensemble of RF + XGBoost)."""
        X_lgd = X[self.lgd_features]

        lgd_rf = self.lgd_rf_model.predict(X_lgd)
        lgd_xgb = self.lgd_xgb_model.predict(X_lgd)

        # Ensemble (average)
        lgd = (lgd_rf + lgd_xgb) / 2
        lgd = np.clip(lgd, 0, 1)  # Clip to valid range

        return lgd

    def predict_ead(self, X):
        """
        Predict Exposure at Default (ensemble of RF + XGBoost).

        The EAD model was trained on standardized EAD values (which equal
        utilization * standardized_loan_amnt). Since utilization ~= 100% for
        defaulted loans, EAD ≈ loan_amnt in standardized form.

        We reverse the standardization to get actual pound amounts.
        """
        X_ead = X[self.ead_features]

        ead_rf = self.ead_rf_model.predict(X_ead)
        ead_xgb = self.ead_xgb_model.predict(X_ead)

        # Ensemble (average) - this gives us standardized EAD
        ead_standardized = (ead_rf + ead_xgb) / 2

        # Reverse standardization to get actual pound amounts
        # Original EAD = (standardized_EAD * std_loan_amnt) + mean_loan_amnt
        # Since EAD was calculated as utilization * loan_amnt (both standardized)
        # and utilization ~= 1.0, the statistics are similar to loan_amnt
        loan_amnt_mean = 14262.275
        loan_amnt_std = 8379.608

        ead_actual = (ead_standardized * loan_amnt_std) + loan_amnt_mean

        # Clip to reasonable range: [0, max_loan_amount]
        ead_actual = np.clip(ead_actual, 0, 35000)

        return ead_actual

    def calculate_expected_loss(self, X):
        """
        Calculate Expected Loss = PD × LGD × EAD

        Args:
            X: DataFrame with loan features

        Returns:
            DataFrame with PD, LGD, EAD, and Expected Loss
        """
        print("\n" + "="*80)
        print("CALCULATING EXPECTED LOSS")
        print("="*80)

        # Predict PD
        print("\n1. Predicting Probability of Default (PD)...")
        pd_probs = self.predict_pd(X)

        # Predict LGD
        print("2. Predicting Loss Given Default (LGD)...")
        lgd_values = self.predict_lgd(X)

        # Predict EAD
        print("3. Predicting Exposure at Default (EAD)...")
        ead_values = self.predict_ead(X)

        # Calculate Expected Loss
        print("4. Calculating Expected Loss...")
        expected_loss = pd_probs * lgd_values * ead_values

        # Create results DataFrame
        results = pd.DataFrame({
            'PD': pd_probs,
            'LGD': lgd_values,
            'EAD': ead_values,
            'Expected_Loss': expected_loss
        })

        print("\n" + "="*80)
        print("EXPECTED LOSS STATISTICS")
        print("="*80)
        print(f"\nPD (Probability of Default):")
        print(f"  Mean: {pd_probs.mean():.2%}")
        print(f"  Median: {np.median(pd_probs):.2%}")

        print(f"\nLGD (Loss Given Default):")
        print(f"  Mean: {lgd_values.mean():.2%}")
        print(f"  Median: {np.median(lgd_values):.2%}")

        print(f"\nEAD (Exposure at Default):")
        print(f"  Mean: £{ead_values.mean():.2f}")
        print(f"  Median: £{np.median(ead_values):.2f}")

        print(f"\nExpected Loss:")
        print(f"  Mean: £{expected_loss.mean():.2f}")
        print(f"  Median: £{np.median(expected_loss):.2f}")
        print(f"  Total: £{expected_loss.sum():,.2f}")

        return results


def main():
    """Main execution: Calculate Expected Loss on test set."""
    print("\n" + "="*80)
    print("EXPECTED LOSS CALCULATION")
    print("="*80)

    # Load test data
    print("\n1. Loading test data...")
    test_path = PROCESSED_DIR / 'test.csv'

    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}")
        return

    test_data = pd.read_csv(test_path)
    print(f"Test data: {len(test_data):,} samples")

    # Store actual defaults if present, then remove from features
    actual_defaults = None
    if 'default' in test_data.columns:
        actual_defaults = test_data['default'].values
        test_features = test_data.drop('default', axis=1)
    else:
        test_features = test_data

    # Initialize calculator
    print("\n2. Initializing Expected Loss Calculator...")
    calculator = ExpectedLossCalculator()
    calculator.load_models()

    # Calculate Expected Loss
    print("\n3. Calculating Expected Loss for test set...")
    results = calculator.calculate_expected_loss(test_features)

    # Add actual default status if available
    if actual_defaults is not None:
        results['Actual_Default'] = actual_defaults

    # Save results
    print("\n4. Saving results...")
    output_path = RESULTS_DIR / 'expected_loss_results.csv'
    results.to_csv(output_path, index=False)
    print(f"[OK] Saved to: {output_path}")

    # Risk segmentation
    print("\n" + "="*80)
    print("RISK SEGMENTATION")
    print("="*80)

    # Segment by Expected Loss (pound amounts)
    # Low: £0-500, Medium: £500-2000, High: £2000-5000, Very High: £5000+
    results['Risk_Category'] = pd.cut(
        results['Expected_Loss'],
        bins=[0, 500, 2000, 5000, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )

    risk_summary = results.groupby('Risk_Category', observed=True).agg({
        'Expected_Loss': ['count', 'mean', 'sum']
    }).round(4)

    print("\n" + risk_summary.to_string())

    print("\n" + "="*80)
    print("[COMPLETE] EXPECTED LOSS CALCULATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
