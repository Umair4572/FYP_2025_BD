"""
Data Preprocessing Module for Credit Risk Assessment FYP
Handles data cleaning, encoding, scaling, and feature transformation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from .config import (
    PREPROCESSING_CONFIG,
    LEAKAGE_COLUMNS,
    TARGET_ENCODING,
    DATASET_CONFIG
)
from .utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.preprocessor')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for credit risk data.

    Performs the following steps:
    1. Target variable creation and filtering
    2. Data leakage prevention
    3. Missing value handling
    4. Outlier detection and treatment
    5. Feature type separation
    6. Categorical encoding
    7. Feature scaling
    8. Data validation
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataPreprocessor.

        Args:
            config: Optional preprocessing configuration dictionary
        """
        self.config = config or PREPROCESSING_CONFIG
        self.target_col = DATASET_CONFIG['target_column']

        # Fitted transformers (will be set during fit())
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        self.scaler = None

        # Feature lists (will be set during fit())
        self.numerical_features = []
        self.categorical_features = []
        self.features_to_drop = []

        # Statistics
        self.missing_stats = {}
        self.outlier_stats = {}

        # Fitted flag
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Learn preprocessing parameters from training data.

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor on training data...")

        df = df.copy()

        # Step 1: Create target variable
        df, target = self._create_target_variable(df)

        # Step 2: Remove data leakage columns
        df = self._remove_leakage_columns(df)

        # Step 3: Identify and drop high-missing columns
        df, self.features_to_drop = self._identify_high_missing_columns(df)
        df = df.drop(columns=self.features_to_drop, errors='ignore')

        # Step 4: Separate numerical and categorical features
        self.numerical_features, self.categorical_features = self._identify_feature_types(df)

        logger.info(f"Identified {len(self.numerical_features)} numerical and "
                   f"{len(self.categorical_features)} categorical features")

        # Step 5: Fit imputers for missing values
        self._fit_imputers(df)

        # Step 6: Fit encoders for categorical features
        df_imputed = self._impute_missing(df.copy())
        self._fit_encoders(df_imputed)

        # Step 7: Fit scaler for numerical features
        df_encoded = self._encode_categorical(df_imputed.copy())
        self._fit_scaler(df_encoded)

        # Step 8: Calculate outlier statistics
        self._calculate_outlier_stats(df)

        self.is_fitted = True
        logger.info("Preprocessor fitting complete")

        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply preprocessing to data.

        Args:
            df: DataFrame to preprocess
            is_training: Whether this is training data (affects target handling)

        Returns:
            Tuple of (preprocessed_df, target) where target is None if not training

        Raises:
            RuntimeError: If preprocessor not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")

        logger.info("Transforming data...")

        df = df.copy()
        target = None

        # Step 1: Handle target variable
        if is_training and self.target_col in df.columns:
            df, target = self._create_target_variable(df)
        elif self.target_col in df.columns:
            df = df.drop(columns=[self.target_col], errors='ignore')

        # Step 2: Remove data leakage columns
        df = self._remove_leakage_columns(df)

        # Step 3: Drop high-missing columns
        df = df.drop(columns=self.features_to_drop, errors='ignore')

        # Step 4: Ensure all expected features are present
        missing_features = set(self.numerical_features + self.categorical_features) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}")
            for feat in missing_features:
                df[feat] = np.nan

        # Step 5: Impute missing values
        df = self._impute_missing(df)

        # Step 6: Encode categorical features
        df = self._encode_categorical(df)

        # Step 7: Handle outliers
        if self.config['handle_outliers']:
            df = self._handle_outliers(df)

        # Step 8: Scale numerical features
        if self.config['scale_features']:
            df = self._scale_features(df)

        # Step 9: Final validation
        df = self._validate_data(df)

        logger.info(f"Transform complete. Shape: {df.shape}")

        return df, target

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Tuple of (preprocessed_df, target)
        """
        self.fit(df)
        return self.transform(df, is_training=True)

    def _create_target_variable(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create binary target variable from loan_status.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame without target, target Series)
        """
        if self.target_col not in df.columns:
            logger.warning(f"Target column '{self.target_col}' not found in data")
            return df, None

        logger.info("Creating binary target variable...")

        # Create binary target
        target = df[self.target_col].copy()

        # Map to binary: 1 = Default, 0 = Non-default
        target_binary = pd.Series(np.nan, index=target.index)

        for status in TARGET_ENCODING['default_class']:
            target_binary[target == status] = 1

        for status in TARGET_ENCODING['non_default_class']:
            target_binary[target == status] = 0

        # Remove rows with uncertain status
        before_count = len(df)
        uncertain_mask = target.isin(TARGET_ENCODING['exclude_class'])
        valid_mask = ~uncertain_mask & target_binary.notna()

        df = df[valid_mask].copy()
        target_binary = target_binary[valid_mask].copy()

        removed_count = before_count - len(df)
        logger.info(f"Removed {removed_count:,} rows with uncertain/excluded status")

        # Print class distribution
        class_counts = target_binary.value_counts()
        logger.info(f"Target distribution:\n  Non-default (0): {class_counts.get(0, 0):,}\n"
                   f"  Default (1): {class_counts.get(1, 0):,}")

        # Remove target from features
        df = df.drop(columns=[self.target_col], errors='ignore')

        return df, target_binary

    def _remove_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that cause data leakage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with leakage columns removed
        """
        leakage_cols_present = [col for col in LEAKAGE_COLUMNS if col in df.columns]

        if leakage_cols_present:
            logger.info(f"Removing {len(leakage_cols_present)} data leakage columns")
            df = df.drop(columns=leakage_cols_present, errors='ignore')

        return df

    def _identify_high_missing_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Identify columns with high missing percentages.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame, list of columns to drop)
        """
        missing_threshold = self.config['missing_threshold']
        missing_pct = (df.isnull().sum() / len(df)) * 100

        high_missing_cols = missing_pct[missing_pct > missing_threshold * 100].index.tolist()

        if high_missing_cols:
            logger.info(f"Identified {len(high_missing_cols)} columns with >{missing_threshold*100}% missing")
            for col in high_missing_cols[:10]:  # Show first 10
                logger.debug(f"  {col}: {missing_pct[col]:.1f}% missing")

        return df, high_missing_cols

    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Separate numerical and categorical features.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        # Numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Categorical features (object, category)
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove ID column if present
        id_col = DATASET_CONFIG.get('id_column')
        if id_col in numerical_features:
            numerical_features.remove(id_col)
        if id_col in categorical_features:
            categorical_features.remove(id_col)

        return numerical_features, categorical_features

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """
        Fit imputers for missing values.

        Args:
            df: Input DataFrame
        """
        logger.info("Fitting imputers for missing values...")

        # Numerical imputer (median strategy)
        if self.numerical_features:
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.numerical_imputer.fit(df[self.numerical_features])

        # Categorical imputer (most frequent strategy)
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(df[self.categorical_features])

        # Store missing statistics
        self.missing_stats = {
            'numerical': df[self.numerical_features].isnull().sum().to_dict() if self.numerical_features else {},
            'categorical': df[self.categorical_features].isnull().sum().to_dict() if self.categorical_features else {}
        }

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted imputers.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        # Impute numerical
        if self.numerical_features and self.numerical_imputer:
            df[self.numerical_features] = self.numerical_imputer.transform(df[self.numerical_features])

        # Impute categorical
        if self.categorical_features and self.categorical_imputer:
            df[self.categorical_features] = self.categorical_imputer.transform(df[self.categorical_features])

        return df

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit label encoders for categorical features.

        Args:
            df: Input DataFrame
        """
        if not self.categorical_features:
            return

        logger.info(f"Fitting encoders for {len(self.categorical_features)} categorical features...")

        for col in self.categorical_features:
            encoder = LabelEncoder()
            # Handle potential string/object types
            encoder.fit(df[col].astype(str))
            self.label_encoders[col] = encoder

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using fitted encoders.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        if not self.categorical_features:
            return df

        for col in self.categorical_features:
            if col in self.label_encoders:
                encoder = self.label_encoders[col]

                # Handle unseen categories
                df[col] = df[col].astype(str)
                unseen_mask = ~df[col].isin(encoder.classes_)

                if unseen_mask.any():
                    logger.warning(f"Found {unseen_mask.sum()} unseen categories in '{col}'")
                    # Assign to most frequent class
                    df.loc[unseen_mask, col] = encoder.classes_[0]

                df[col] = encoder.transform(df[col])

        return df

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fit scaler for numerical features.

        Args:
            df: Input DataFrame
        """
        if not self.numerical_features or not self.config['scale_features']:
            return

        logger.info("Fitting scaler for numerical features...")
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.numerical_features])

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using fitted scaler.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with scaled features
        """
        if not self.numerical_features or not self.scaler:
            return df

        df[self.numerical_features] = self.scaler.transform(df[self.numerical_features])

        return df

    def _calculate_outlier_stats(self, df: pd.DataFrame) -> None:
        """
        Calculate outlier statistics for numerical features.

        Args:
            df: Input DataFrame
        """
        if not self.config['handle_outliers']:
            return

        method = self.config['outlier_method']
        threshold = self.config['outlier_threshold']

        logger.info(f"Calculating outlier statistics (method={method})...")

        for col in self.numerical_features:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                self.outlier_stats[col] = {
                    'method': 'iqr',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

                self.outlier_stats[col] = {
                    'method': 'zscore',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers by capping (not removing).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with capped outliers
        """
        if not self.outlier_stats:
            return df

        for col, bounds in self.outlier_stats.items():
            if col not in df.columns:
                continue

            lower = bounds['lower_bound']
            upper = bounds['upper_bound']

            # Cap outliers
            df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform final data validation.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        # Check for infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())

        if inf_cols:
            logger.warning(f"Replaced infinite values in {len(inf_cols)} columns")

        # Check for remaining NaNs
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            logger.warning(f"Found {nan_counts.sum()} remaining NaN values")
            # Fill with column median/mode
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in [np.number]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted preprocessor to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        save_pickle(self, filepath)
        logger.info(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'DataPreprocessor':
        """
        Load fitted preprocessor from file.

        Args:
            filepath: Path to saved preprocessor

        Returns:
            Loaded DataPreprocessor instance
        """
        preprocessor = load_pickle(filepath)
        logger.info(f"Loaded preprocessor from {filepath}")
        return preprocessor

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after preprocessing.

        Returns:
            List of feature names
        """
        return self.numerical_features + self.categorical_features
