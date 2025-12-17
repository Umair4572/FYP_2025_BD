"""
Data Resampling Module for Credit Risk Assessment

This module implements various resampling techniques to handle class imbalance,
with a focus on SMOTE (Synthetic Minority Over-sampling Technique) and its variants.
"""

import logging
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


class DataResampler:
    """
    Handles data resampling to address class imbalance.

    Supports multiple resampling strategies including SMOTE variants
    and combined over/under-sampling methods.
    """

    def __init__(
        self,
        strategy: str = 'smote',
        sampling_ratio: float = 0.6,
        random_state: int = 42,
        k_neighbors: int = 5,
        categorical_features: Optional[list] = None
    ):
        """
        Initialize the DataResampler.

        Args:
            strategy: Resampling strategy to use
                - 'smote': Standard SMOTE
                - 'smotenc': SMOTE for mixed categorical/numerical data
                - 'borderline': Borderline-SMOTE (focuses on borderline cases)
                - 'svm': SVM-SMOTE (uses SVM to identify support vectors)
                - 'adasyn': ADASYN (adaptive synthetic sampling)
                - 'smote_tomek': SMOTE + Tomek links cleaning
                - 'smote_enn': SMOTE + Edited Nearest Neighbors cleaning
            sampling_ratio: Desired ratio of minority to majority class after resampling
                - 0.5: Fully balanced (50:50)
                - 0.6: 60% minority, 40% majority (recommended)
                - 0.7: 70% minority, 30% majority
            random_state: Random seed for reproducibility
            k_neighbors: Number of nearest neighbors for SMOTE
            categorical_features: Indices of categorical features for SMOTENC
        """
        self.strategy = strategy.lower()
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.categorical_features = categorical_features
        self.resampler = None
        self._create_resampler()

    def _create_resampler(self):
        """Create the appropriate resampler based on strategy."""
        if self.strategy == 'smote':
            self.resampler = SMOTE(
                sampling_strategy=self.sampling_ratio,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )

        elif self.strategy == 'smotenc':
            if self.categorical_features is None:
                raise ValueError("categorical_features must be provided for SMOTENC")
            self.resampler = SMOTENC(
                categorical_features=self.categorical_features,
                sampling_strategy=self.sampling_ratio,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )

        elif self.strategy == 'borderline':
            self.resampler = BorderlineSMOTE(
                sampling_strategy=self.sampling_ratio,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )

        elif self.strategy == 'svm':
            self.resampler = SVMSMOTE(
                sampling_strategy=self.sampling_ratio,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )

        elif self.strategy == 'adasyn':
            self.resampler = ADASYN(
                sampling_strategy=self.sampling_ratio,
                n_neighbors=self.k_neighbors,
                random_state=self.random_state
            )

        elif self.strategy == 'smote_tomek':
            self.resampler = SMOTETomek(
                sampling_strategy=self.sampling_ratio,
                random_state=self.random_state,
                smote=SMOTE(
                    k_neighbors=self.k_neighbors,
                    random_state=self.random_state
                )
            )

        elif self.strategy == 'smote_enn':
            self.resampler = SMOTEENN(
                sampling_strategy=self.sampling_ratio,
                random_state=self.random_state,
                smote=SMOTE(
                    k_neighbors=self.k_neighbors,
                    random_state=self.random_state
                )
            )

        else:
            raise ValueError(
                f"Unknown resampling strategy: {self.strategy}. "
                f"Choose from: smote, smotenc, borderline, svm, adasyn, smote_tomek, smote_enn"
            )

    def fit_resample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Apply resampling to the training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Store original types
        is_dataframe = isinstance(X, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        # Store column names and index for reconstruction
        if is_dataframe:
            columns = X.columns
            X_array = X.values
        else:
            X_array = X
            columns = None

        if is_series:
            y_array = y.values
            y_name = y.name
        else:
            y_array = y
            y_name = None

        # Get original class distribution
        unique, counts = np.unique(y_array, return_counts=True)
        class_dist_before = dict(zip(unique, counts))

        logger.info(f"Applying {self.strategy.upper()} resampling...")
        logger.info(f"Original class distribution: {class_dist_before}")
        logger.info(f"Target sampling ratio: {self.sampling_ratio}")

        # Apply resampling
        try:
            X_resampled, y_resampled = self.resampler.fit_resample(X_array, y_array)
        except Exception as e:
            logger.error(f"Resampling failed: {str(e)}")
            logger.warning("Returning original data without resampling")
            return X, y

        # Get new class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_dist_after = dict(zip(unique, counts))

        logger.info(f"Resampled class distribution: {class_dist_after}")
        logger.info(
            f"Generated {len(y_resampled) - len(y_array)} synthetic samples "
            f"({len(y_resampled)} total from {len(y_array)})"
        )

        # Convert back to original types
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=columns)

        if is_series:
            y_resampled = pd.Series(y_resampled, name=y_name)

        return X_resampled, y_resampled

    def get_sampling_info(self) -> dict:
        """
        Get information about the resampling configuration.

        Returns:
            Dictionary with resampling configuration details
        """
        return {
            'strategy': self.strategy,
            'sampling_ratio': self.sampling_ratio,
            'k_neighbors': self.k_neighbors,
            'random_state': self.random_state
        }


def apply_smote_to_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_ratio: float = 0.6,
    strategy: str = 'smote',
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to apply SMOTE to training data.

    Args:
        X_train: Training features
        y_train: Training targets
        sampling_ratio: Desired minority to majority ratio
        strategy: Resampling strategy
        k_neighbors: Number of neighbors for SMOTE
        random_state: Random seed

    Returns:
        Tuple of (X_train_resampled, y_train_resampled)
    """
    resampler = DataResampler(
        strategy=strategy,
        sampling_ratio=sampling_ratio,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    return resampler.fit_resample(X_train, y_train)
