"""
EAD (Exposure at Default) Regression Model Training
Trains Random Forest and XGBoost regressors to predict EAD for defaulted loans
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Random state
RANDOM_STATE = 42

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data'
EAD_DIR = DATA_DIR / 'ead'
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'


def load_ead_data():
    """Load EAD data for training."""
    print("\n" + "="*80)
    print("LOADING EAD DATA")
    print("="*80)

    # Load defaulted loans with EAD
    train_path = EAD_DIR / 'defaulted_loans_ead.csv'

    if not train_path.exists():
        print(f"ERROR: EAD data not found at {train_path}")
        print("Please run EAD simulation first: python run_ead_simulation.py")
        return None, None, None

    df = pd.read_csv(train_path)
    print(f"\nLoaded {len(df):,} defaulted loans with EAD")

    # Remove rows with NaN EAD
    df = df.dropna(subset=['ead'])
    print(f"After removing NaN: {len(df):,} samples")

    # Separate features and target
    target_col = 'ead'
    exclude_cols = ['ead', 'ead_utilization', 'default', 'lgd']  # Exclude target and related columns

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Target: {target_col}")
    print(f"EAD range: £{y.min():.2f} to £{y.max():.2f}")
    print(f"EAD mean: £{y.mean():.2f}")

    return X, y, feature_cols


def train_test_split_temporal(X, y, test_size=0.2):
    """Split data temporally (last 20% as test)."""
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE with protection against division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    print(f"\n{model_name} Results:")
    print(f"  MAE:  £{mae:.4f}")
    print(f"  RMSE: £{rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest regressor."""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST REGRESSOR")
    print("="*80)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    # Predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Clip predictions to non-negative
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)

    # Metrics
    print("\n--- Training Set ---")
    train_metrics = calculate_metrics(y_train, y_pred_train, "Random Forest")

    print("\n--- Test Set ---")
    test_metrics = calculate_metrics(y_test, y_pred_test, "Random Forest")

    return model, train_metrics, test_metrics, y_pred_test


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost regressor."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST REGRESSOR")
    print("="*80)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='rmse'
    )

    print("\nTraining...")
    model.fit(X_train, y_train, verbose=False)

    # Predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Clip predictions to non-negative
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)

    # Metrics
    print("\n--- Training Set ---")
    train_metrics = calculate_metrics(y_train, y_pred_train, "XGBoost")

    print("\n--- Test Set ---")
    test_metrics = calculate_metrics(y_test, y_pred_test, "XGBoost")

    return model, train_metrics, test_metrics, y_pred_test


def create_ensemble(rf_pred, xgb_pred, y_test):
    """Create ensemble by averaging RF and XGBoost predictions."""
    print("\n" + "="*80)
    print("CREATING ENSEMBLE (AVERAGE)")
    print("="*80)

    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_pred = np.maximum(ensemble_pred, 0)

    print("\n--- Test Set ---")
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred, "Ensemble")

    return ensemble_pred, ensemble_metrics


def save_results(rf_model, xgb_model, rf_metrics, xgb_metrics, ensemble_metrics, feature_cols):
    """Save models and results."""
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)

    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save Random Forest model
    rf_path = MODELS_DIR / 'ead_random_forest.pkl'
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Random Forest saved to: {rf_path}")

    # Save XGBoost model
    xgb_path = MODELS_DIR / 'ead_xgboost.pkl'
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"✓ XGBoost saved to: {xgb_path}")

    # Save metrics
    metrics_data = {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics,
        'ensemble': ensemble_metrics,
        'feature_columns': feature_cols
    }

    metrics_path = RESULTS_DIR / 'ead_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_data, f)
    print(f"✓ Metrics saved to: {metrics_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("PHASE 4: EAD MODEL TRAINING")
    print("="*80)

    # 1. Load data
    print("\n1. Loading EAD data...")
    X, y, feature_cols = load_ead_data()

    if X is None:
        return

    # 2. Split data
    print("\n2. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # 3. Train Random Forest
    print("\n3. Training Random Forest...")
    rf_model, rf_train_metrics, rf_test_metrics, rf_pred = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # 4. Train XGBoost
    print("\n4. Training XGBoost...")
    xgb_model, xgb_train_metrics, xgb_test_metrics, xgb_pred = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # 5. Create ensemble
    print("\n5. Creating ensemble...")
    ensemble_pred, ensemble_metrics = create_ensemble(rf_pred, xgb_pred, y_test)

    # 6. Save results
    print("\n6. Saving results...")
    save_results(rf_model, xgb_model, rf_test_metrics, xgb_test_metrics,
                ensemble_metrics, feature_cols)

    # 7. Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    comparison = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Ensemble'],
        'MAE': [rf_test_metrics['mae'], xgb_test_metrics['mae'], ensemble_metrics['mae']],
        'RMSE': [rf_test_metrics['rmse'], xgb_test_metrics['rmse'], ensemble_metrics['rmse']],
        'R²': [rf_test_metrics['r2'], xgb_test_metrics['r2'], ensemble_metrics['r2']],
        'MAPE (%)': [rf_test_metrics['mape'], xgb_test_metrics['mape'], ensemble_metrics['mape']]
    })

    print("\n" + comparison.to_string(index=False))

    # Best model
    best_idx = comparison['R²'].idxmax()
    best_model = comparison.loc[best_idx, 'Model']

    print(f"\n🏆 BEST MODEL: {best_model} (R² = {comparison.loc[best_idx, 'R²']:.4f})")

    print("\n" + "="*80)
    print("✅ EAD MODEL TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
