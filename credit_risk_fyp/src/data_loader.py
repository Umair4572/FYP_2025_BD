"""
Data Loading Module for Credit Risk Assessment FYP
Provides optimized data loading with memory management and chunk processing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DATA_LOADING_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR
from .utils import optimize_memory

# Setup logger
logger = logging.getLogger('credit_risk_fyp.data_loader')


class DataLoader:
    """
    Optimized data loader for large datasets with memory management.

    Features:
    - Chunk-based loading for large files
    - Automatic dtype optimization
    - Memory usage tracking
    - Support for multiple file formats (CSV, Parquet)
    - Progress bars for user feedback
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        optimize_dtypes: bool = True,
        verbose: bool = True
    ):
        """
        Initialize DataLoader.

        Args:
            chunk_size: Number of rows to load at once (None = load all)
            optimize_dtypes: Whether to optimize data types for memory
            verbose: Print progress and memory information
        """
        self.chunk_size = chunk_size or DATA_LOADING_CONFIG['chunk_size']
        self.optimize_dtypes = optimize_dtypes
        self.verbose = verbose

    def load_dataset(
        self,
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset from file with optimization.

        Args:
            filepath: Path to data file
            **kwargs: Additional arguments for pd.read_csv/pd.read_parquet

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading dataset from: {filepath}")

        # Determine file format
        if filepath.suffix.lower() == '.csv':
            df = self._load_csv(filepath, **kwargs)
        elif filepath.suffix.lower() in ['.parquet', '.pq']:
            df = self._load_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Optimize dtypes if requested
        if self.optimize_dtypes:
            if self.verbose:
                print("\nOptimizing data types...")
            df = optimize_memory(df, verbose=self.verbose)

        # Report final memory usage
        if self.verbose:
            self._report_memory_usage(df)

        logger.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")

        return df

    def _load_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with chunk processing if needed.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        # Get file size for progress estimation
        file_size = filepath.stat().st_size / (1024 ** 3)  # Convert to GB

        if self.verbose:
            print(f"\nFile size: {file_size:.2f} GB")

        # If file is large, use chunked reading
        if file_size > 1.0 and 'nrows' not in kwargs:  # If file > 1GB and not limited
            logger.info("Large file detected. Using chunked reading...")
            return self._load_csv_chunked(filepath, **kwargs)
        else:
            # Load entire file at once
            df = pd.read_csv(filepath, **kwargs)
            if self.verbose:
                print(f"Loaded {len(df):,} rows")
            return df

    def _load_csv_chunked(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Load large CSV file in chunks.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Concatenated DataFrame
        """
        chunks = []

        # First, get total number of rows for progress bar
        if self.verbose:
            print("Counting rows...")
            with open(filepath) as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header

        # Read in chunks
        chunk_iter = pd.read_csv(filepath, chunksize=self.chunk_size, **kwargs)

        if self.verbose:
            num_chunks = (total_rows // self.chunk_size) + 1
            chunk_iter = tqdm(chunk_iter, total=num_chunks, desc="Loading chunks")

        for chunk in chunk_iter:
            if self.optimize_dtypes:
                chunk = optimize_memory(chunk, verbose=False)
            chunks.append(chunk)

        logger.info(f"Concatenating {len(chunks)} chunks...")
        df = pd.concat(chunks, ignore_index=True)

        return df

    def _load_parquet(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Load Parquet file (more memory efficient than CSV).

        Args:
            filepath: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            Loaded DataFrame
        """
        df = pd.read_parquet(filepath, **kwargs)

        if self.verbose:
            print(f"Loaded {len(df):,} rows from Parquet file")

        return df

    def load_multiple_datasets(
        self,
        filepaths: List[Union[str, Path]],
        concat_axis: int = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple datasets.

        Args:
            filepaths: List of file paths to load
            concat_axis: Axis to concatenate (0=rows, 1=columns)
            **kwargs: Additional arguments for loading

        Returns:
            Concatenated DataFrame
        """
        if not filepaths:
            raise ValueError("No filepaths provided")

        logger.info(f"Loading {len(filepaths)} datasets...")

        datasets = []
        for filepath in tqdm(filepaths, desc="Loading files", disable=not self.verbose):
            df = self.load_dataset(filepath, **kwargs)
            datasets.append(df)

        logger.info("Concatenating datasets...")
        combined_df = pd.concat(datasets, axis=concat_axis, ignore_index=True)

        if self.verbose:
            print(f"\nCombined dataset:")
            print(f"  Rows: {len(combined_df):,}")
            print(f"  Columns: {len(combined_df.columns)}")
            self._report_memory_usage(combined_df)

        return combined_df

    @staticmethod
    def _report_memory_usage(df: pd.DataFrame) -> None:
        """
        Print memory usage summary.

        Args:
            df: DataFrame to analyze
        """
        mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"\nMemory usage: {mem_usage:.2f} MB")

        # Per column memory usage (top 10)
        col_mem = df.memory_usage(deep=True).sort_values(ascending=False)
        print("\nTop 10 memory-consuming columns:")
        for col, mem in col_mem.head(10).items():
            print(f"  {col:30s}: {mem / (1024**2):8.2f} MB")

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform basic data quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating dataset...")

        validation_results = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'columns_all_null': df.columns[df.isnull().all()].tolist(),
            'columns_all_same': df.columns[df.nunique() == 1].tolist(),
            'infinite_values': {},
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 ** 2)
        }

        # Check for infinite values in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validation_results['infinite_values'][col] = inf_count

        # Print summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("DATASET VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Rows: {validation_results['n_rows']:,}")
            print(f"Columns: {validation_results['n_cols']}")
            print(f"Duplicate rows: {validation_results['duplicate_rows']:,}")
            print(f"Columns with all NULL: {len(validation_results['columns_all_null'])}")
            print(f"Columns with same value: {len(validation_results['columns_all_same'])}")
            print(f"Columns with infinite values: {len(validation_results['infinite_values'])}")
            print(f"Memory usage: {validation_results['memory_mb']:.2f} MB")
            print("=" * 60)

            if validation_results['duplicate_rows'] > 0:
                print(f"⚠ Warning: Found {validation_results['duplicate_rows']:,} duplicate rows")

            if validation_results['columns_all_null']:
                print(f"⚠ Warning: {len(validation_results['columns_all_null'])} columns are completely NULL")

            if validation_results['infinite_values']:
                print(f"⚠ Warning: Found infinite values in {len(validation_results['infinite_values'])} columns")

        return validation_results

    def save_dataset(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = 'csv',
        **kwargs
    ) -> None:
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            filepath: Path to save file
            format: File format ('csv' or 'parquet')
            **kwargs: Additional arguments for saving
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dataset to: {filepath}")

        if format == 'csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if self.verbose:
            file_size = filepath.stat().st_size / (1024 ** 2)
            print(f"✓ Saved {len(df):,} rows to {filepath} ({file_size:.2f} MB)")


def load_train_test_split(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    optimize: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load train, validation, and test sets.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        optimize: Whether to optimize memory

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = DataLoader(optimize_dtypes=optimize, verbose=True)

    train_df = loader.load_dataset(train_path) if train_path else None
    val_df = loader.load_dataset(val_path) if val_path else None
    test_df = loader.load_dataset(test_path) if test_path else None

    return train_df, val_df, test_df


# Convenience function
def load_data(filepath: Union[str, Path], optimize: bool = True, **kwargs) -> pd.DataFrame:
    """
    Quick function to load a dataset.

    Args:
        filepath: Path to data file
        optimize: Whether to optimize memory
        **kwargs: Additional arguments

    Returns:
        Loaded DataFrame
    """
    loader = DataLoader(optimize_dtypes=optimize, verbose=True)
    return loader.load_dataset(filepath, **kwargs)


if __name__ == "__main__":
    """
    Run this module directly to view dataset information.

    Usage:
        python -m credit_risk_fyp.src.data_loader train  # View training data
        python -m credit_risk_fyp.src.data_loader test   # View test data
    """
    import sys
    from .config import DATASET_CONFIG

    # Default to showing both if no argument
    show_train = True
    show_test = True

    if len(sys.argv) > 1:
        dataset_choice = sys.argv[1].lower()
        if dataset_choice == 'train':
            show_test = False
        elif dataset_choice == 'test':
            show_train = False
        else:
            print(f"Unknown option: {dataset_choice}")
            print("Usage: python -m credit_risk_fyp.src.data_loader [train|test]")
            sys.exit(1)

    print("=" * 80)
    print("CREDIT RISK FYP - DATASET VIEWER")
    print("=" * 80)

    loader = DataLoader(optimize_dtypes=True, verbose=True)

    # Load and display training data
    if show_train:
        print("\n" + "=" * 80)
        print("TRAINING DATASET")
        print("=" * 80)

        train_path = RAW_DATA_DIR / DATASET_CONFIG['train_dataset']
        print(f"\nLoading from: {train_path}")

        try:
            train_df = loader.load_dataset(train_path)

            print("\n" + "-" * 80)
            print("DATASET INFORMATION")
            print("-" * 80)
            print(f"Shape: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
            print(f"Memory usage: {train_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

            # Target column info
            target_col = DATASET_CONFIG['target_column']
            if target_col in train_df.columns:
                print(f"\nTarget column: '{target_col}'")
                print(f"Target distribution:")
                value_counts = train_df[target_col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(train_df)) * 100
                    print(f"  {val}: {count:,} ({pct:.2f}%)")
                print(f"Missing values: {train_df[target_col].isna().sum():,}")

            print(f"\n" + "-" * 80)
            print("FIRST 5 ROWS")
            print("-" * 80)
            print(train_df.head())

            print(f"\n" + "-" * 80)
            print("COLUMN INFO")
            print("-" * 80)
            print(train_df.info())

            print(f"\n" + "-" * 80)
            print("MISSING VALUES")
            print("-" * 80)
            missing = train_df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                print(f"\nColumns with missing values (top 20):")
                for col, count in missing.head(20).items():
                    pct = (count / len(train_df)) * 100
                    print(f"  {col:40s}: {count:8,} ({pct:6.2f}%)")
            else:
                print("No missing values found!")

            print(f"\n" + "-" * 80)
            print("BASIC STATISTICS")
            print("-" * 80)
            print(train_df.describe())

        except Exception as e:
            print(f"\n[ERROR] Failed to load training data: {e}")

    # Load and display test data
    if show_test:
        print("\n" + "=" * 80)
        print("TEST DATASET")
        print("=" * 80)

        test_path = RAW_DATA_DIR / DATASET_CONFIG['test_dataset']
        print(f"\nLoading from: {test_path}")

        try:
            test_df = loader.load_dataset(test_path)

            print("\n" + "-" * 80)
            print("DATASET INFORMATION")
            print("-" * 80)
            print(f"Shape: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
            print(f"Memory usage: {test_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

            # Check if target column exists
            target_col = DATASET_CONFIG['target_column']
            if target_col in test_df.columns:
                print(f"\nTarget column: '{target_col}'")
                print(f"Target distribution:")
                value_counts = test_df[target_col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(test_df)) * 100
                    print(f"  {val}: {count:,} ({pct:.2f}%)")
                print(f"Missing values: {test_df[target_col].isna().sum():,}")
            else:
                print(f"\n[NOTE] Target column '{target_col}' not found in test dataset")

            print(f"\n" + "-" * 80)
            print("FIRST 5 ROWS")
            print("-" * 80)
            print(test_df.head())

            print(f"\n" + "-" * 80)
            print("COLUMN INFO")
            print("-" * 80)
            print(test_df.info())

            print(f"\n" + "-" * 80)
            print("MISSING VALUES")
            print("-" * 80)
            missing = test_df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                print(f"\nColumns with missing values (top 20):")
                for col, count in missing.head(20).items():
                    pct = (count / len(test_df)) * 100
                    print(f"  {col:40s}: {count:8,} ({pct:6.2f}%)")
            else:
                print("No missing values found!")

            print(f"\n" + "-" * 80)
            print("BASIC STATISTICS")
            print("-" * 80)
            print(test_df.describe())

        except Exception as e:
            print(f"\n[ERROR] Failed to load test data: {e}")

    print("\n" + "=" * 80)
    print("DATASET VIEWER COMPLETE")
    print("=" * 80)
