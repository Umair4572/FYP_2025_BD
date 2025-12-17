"""
Utility functions for Credit Risk Assessment FYP
Contains helper functions for GPU setup, logging, data processing, and visualization
"""

import os
import logging
import random
import pickle
import joblib
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import LOGGING_CONFIG, GPU_CONFIG, VISUALIZATION_CONFIG


def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """
    Setup logging configuration for the project.

    Args:
        config: Optional logging configuration dictionary

    Returns:
        Configured logger instance
    """
    if config is None:
        config = LOGGING_CONFIG

    # Create logger
    logger = logging.getLogger('credit_risk_fyp')
    logger.setLevel(getattr(logging, config['level']))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(config['format'])

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config['level']))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if config.get('log_to_file', False):
        log_file = config['log_file']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, config['level']))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_gpu() -> None:
    """
    Configure GPU for optimal performance with TensorFlow and gradient boosting libraries.
    Sets up memory growth, mixed precision, and GPU device visibility.
    """
    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)

    # TensorFlow GPU setup
    try:
        import tensorflow as tf

        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')

        if gpus and GPU_CONFIG['use_gpu']:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  [{i}] {gpu.name}")

            try:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                if GPU_CONFIG['memory_growth']:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("✓ Memory growth enabled for TensorFlow")

                # Set visible devices
                if GPU_CONFIG['gpu_id'] is not None:
                    tf.config.set_visible_devices(gpus[GPU_CONFIG['gpu_id']], 'GPU')
                    print(f"✓ Using GPU {GPU_CONFIG['gpu_id']}")

                # Enable mixed precision for faster training
                if GPU_CONFIG['mixed_precision']:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    print("✓ Mixed precision (float16) enabled")

            except RuntimeError as e:
                print(f"✗ GPU configuration error: {e}")

        elif not gpus:
            print("⚠ No GPU found - using CPU only")
            print("  For optimal performance, install CUDA and cuDNN")
        else:
            print("⚠ GPU available but disabled in config")

    except ImportError:
        print("⚠ TensorFlow not installed - skipping TensorFlow GPU setup")

    # XGBoost GPU check
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")
    except ImportError:
        print("⚠ XGBoost not installed")

    # LightGBM GPU check
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM version: {lgb.__version__}")
    except ImportError:
        print("⚠ LightGBM not installed")

    # CatBoost GPU check
    try:
        import catboost as cb
        print(f"✓ CatBoost version: {cb.__version__}")
    except ImportError:
        print("⚠ CatBoost not installed")

    print("=" * 80)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"✓ Random seeds set to {seed}")


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        y: Target variable array

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    print(f"Class weights calculated: {class_weights}")
    return class_weights


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Saved object to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    print(f"✓ Loaded object from {filepath}")
    return obj


def save_joblib(obj: Any, filepath: Union[str, Path], compress: int = 3) -> None:
    """
    Save object using joblib (better for large numpy arrays).

    Args:
        obj: Object to save
        filepath: Path to save file
        compress: Compression level (0-9)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, filepath, compress=compress)
    print(f"✓ Saved object to {filepath}")


def load_joblib(filepath: Union[str, Path]) -> Any:
    """
    Load object using joblib.

    Args:
        filepath: Path to joblib file

    Returns:
        Loaded object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    obj = joblib.load(filepath)
    print(f"✓ Loaded object from {filepath}")
    return obj


def print_memory_usage(df: Optional[pd.DataFrame] = None) -> None:
    """
    Print current memory usage or DataFrame memory usage.

    Args:
        df: Optional DataFrame to check memory usage
    """
    import psutil

    # System memory
    memory = psutil.virtual_memory()
    print(f"\nSystem Memory Usage:")
    print(f"  Total: {memory.total / (1024**3):.2f} GB")
    print(f"  Available: {memory.available / (1024**3):.2f} GB")
    print(f"  Used: {memory.used / (1024**3):.2f} GB ({memory.percent}%)")

    # DataFrame memory
    if df is not None:
        mem_usage = df.memory_usage(deep=True).sum()
        print(f"\nDataFrame Memory Usage:")
        print(f"  Total: {mem_usage / (1024**2):.2f} MB")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory per row: {mem_usage / len(df):.2f} bytes")


def plot_correlation_matrix(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame to analyze
        target_col: Optional target column to highlight correlations
        figsize: Figure size tuple
        save_path: Optional path to save figure
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    # Create figure
    plt.figure(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Saved correlation matrix to {save_path}")

    plt.show()

    # If target column specified, print top correlations
    if target_col and target_col in corr_matrix.columns:
        print(f"\nTop 10 features correlated with {target_col}:")
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(target_corr.head(11))  # Top 10 + target itself


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot distribution histograms for numerical columns.

    Args:
        df: DataFrame to plot
        columns: Optional list of columns to plot
        figsize: Figure size tuple
        save_path: Optional path to save figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx < len(axes):
            df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')

    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Saved distributions plot to {save_path}")

    plt.show()


def detect_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 3.0
) -> Dict[str, np.ndarray]:
    """
    Detect outliers in numerical columns.

    Args:
        df: DataFrame to analyze
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Dictionary mapping column names to boolean arrays indicating outliers
    """
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

    # Print summary
    print("\nOutlier Detection Summary:")
    print(f"Method: {method}, Threshold: {threshold}")
    print("-" * 60)
    for col, is_outlier in outliers.items():
        n_outliers = is_outlier.sum()
        pct_outliers = (n_outliers / len(df)) * 100
        if n_outliers > 0:
            print(f"{col:30s}: {n_outliers:6d} ({pct_outliers:5.2f}%)")

    return outliers


def calculate_vif(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

    Args:
        df: DataFrame with numerical features
        threshold: VIF threshold (values > threshold indicate multicollinearity)

    Returns:
        DataFrame with features and their VIF values
    """
    numerical_df = df.select_dtypes(include=[np.number])

    # Remove columns with zero variance
    numerical_df = numerical_df.loc[:, numerical_df.std() > 0]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = numerical_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numerical_df.values, i)
        for i in range(len(numerical_df.columns))
    ]

    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("\nVariance Inflation Factor (VIF) Analysis:")
    print(f"Threshold: {threshold}")
    print("-" * 60)
    print(vif_data.to_string(index=False))

    high_vif = vif_data[vif_data['VIF'] > threshold]
    if len(high_vif) > 0:
        print(f"\n⚠ {len(high_vif)} features with VIF > {threshold} (potential multicollinearity)")

    return vif_data


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive statistics for all features.

    Args:
        df: DataFrame to analyze

    Returns:
        DataFrame with feature statistics
    """
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'unique_count': df.nunique(),
        'unique_pct': (df.nunique() / len(df)) * 100
    })

    # Add numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        stats.loc[col, 'mean'] = df[col].mean()
        stats.loc[col, 'std'] = df[col].std()
        stats.loc[col, 'min'] = df[col].min()
        stats.loc[col, 'max'] = df[col].max()
        stats.loc[col, 'median'] = df[col].median()
        stats.loc[col, 'skewness'] = df[col].skew()
        stats.loc[col, 'kurtosis'] = df[col].kurtosis()

    return stats


def export_to_excel(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    filepath: Union[str, Path],
    sheet_name: str = 'Sheet1'
) -> None:
    """
    Export DataFrame(s) to Excel file.

    Args:
        data: DataFrame or dictionary of DataFrames
        filepath: Path to save Excel file
        sheet_name: Sheet name (if single DataFrame)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        if isinstance(data, dict):
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=True)
        else:
            data.to_excel(writer, sheet_name=sheet_name, index=True)

    print(f"✓ Exported data to {filepath}")


def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numerical dtypes.

    Args:
        df: DataFrame to optimize
        verbose: Print memory savings

    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f'Memory usage before optimization: {start_mem:.2f} MB')
        print(f'Memory usage after optimization: {end_mem:.2f} MB')
        print(f'Memory decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df
