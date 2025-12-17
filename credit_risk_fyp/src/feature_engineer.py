"""
Feature Engineering Module for Credit Risk Assessment FYP
Creates derived features, ratios, interactions, and aggregations
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

from .config import FEATURE_ENGINEERING_CONFIG
from .utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.feature_engineer')


class FeatureEngineer:
    """
    Comprehensive feature engineering for credit risk data.

    Creates:
    - Financial ratios (loan-to-income, installment-to-income, etc.)
    - Credit behavior indicators (delinquency score, account diversity)
    - Interaction features (int_rate × dti, fico × dti)
    - Time-based features (credit age, loan season)
    - Aggregation features
    - Binned/discretized features
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: Optional feature engineering configuration
        """
        self.config = config or FEATURE_ENGINEERING_CONFIG
        self.feature_names = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn parameters for feature engineering (e.g., quantile bins).

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature engineer...")

        # Calculate quantiles for binning if needed
        self.bin_edges = {}

        if self.config['binning_strategy'] == 'quantile':
            n_bins = self.config['n_bins']

            # Income bins
            if 'annual_inc' in df.columns:
                self.bin_edges['income'] = df['annual_inc'].quantile(
                    np.linspace(0, 1, n_bins + 1)
                ).values

            # Loan amount bins
            if 'loan_amnt' in df.columns:
                self.bin_edges['loan_amnt'] = df['loan_amnt'].quantile(
                    np.linspace(0, 1, n_bins + 1)
                ).values

        self.is_fitted = True
        logger.info("Feature engineer fitting complete")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self.is_fitted:
            logger.warning("Feature engineer not fitted. Fitting on current data...")
            self.fit(df)

        logger.info("Engineering features...")

        df = df.copy()
        initial_features = len(df.columns)

        # Create different types of features based on config
        if self.config['create_ratios']:
            df = self.create_ratio_features(df)

        if self.config['create_time_features']:
            df = self.create_time_features(df)

        df = self.create_credit_features(df)

        # Create risk indicator features (always applied for better default prediction)
        df = self.create_risk_indicators(df)

        if self.config['create_interactions']:
            df = self.create_interaction_features(df)

        if self.config['create_aggregations']:
            df = self.create_aggregation_features(df)

        # Binning
        df = self.create_binned_features(df)

        final_features = len(df.columns)
        new_features = final_features - initial_features

        logger.info(f"Created {new_features} new features (total: {final_features})")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        self.fit(df)
        return self.transform(df)

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratio features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ratio features added
        """
        logger.info("Creating ratio features...")

        # Loan to income ratio
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

        # Installment to income ratio
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['installment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)

        # DTI ratio normalized
        if 'dti' in df.columns:
            df['dti_ratio'] = df['dti'] / 100.0

        # Credit utilization
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            df['credit_utilization'] = df['revol_util'] / 100.0

        # Payment coverage (can borrower afford monthly payment?)
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_coverage'] = df['installment'] / ((df['annual_inc'] / 12) + 1)

        # Revolving balance to income
        if 'revol_bal' in df.columns and 'annual_inc' in df.columns:
            df['revol_bal_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)

        # Funded amount ratio
        if 'funded_amnt' in df.columns and 'loan_amnt' in df.columns:
            df['funded_ratio'] = df['funded_amnt'] / (df['loan_amnt'] + 1)

        return df

    def create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit behavior indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with credit features added
        """
        logger.info("Creating credit behavior features...")

        # Account diversity (open vs total accounts)
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            df['account_diversity'] = df['open_acc'] / (df['total_acc'] + 1)

        # Recent inquiry rate
        if 'inq_last_6mths' in df.columns:
            df['recent_inquiry_rate'] = df['inq_last_6mths'] / 6.0

        # Delinquency score (weighted sum of negative events)
        delinq_score = 0
        if 'delinq_2yrs' in df.columns:
            delinq_score += df['delinq_2yrs'] * 2
        if 'pub_rec' in df.columns:
            delinq_score += df['pub_rec'] * 3
        if 'pub_rec_bankruptcies' in df.columns:
            delinq_score += df['pub_rec_bankruptcies'] * 5

        df['delinquency_score'] = delinq_score

        # Total number of accounts
        if 'total_acc' in df.columns and 'open_acc' in df.columns:
            df['closed_acc'] = df['total_acc'] - df['open_acc']

        # Average account balance
        if 'total_bal_ex_mort' in df.columns and 'total_acc' in df.columns:
            df['avg_account_balance'] = df['total_bal_ex_mort'] / (df['total_acc'] + 1)

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with time features added
        """
        logger.info("Creating time-based features...")

        # Credit age
        if 'earliest_cr_line' in df.columns:
            # Try to parse dates
            try:
                earliest_dates = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
                reference_date = pd.Timestamp('2020-12-31')  # Use dataset end date

                # Calculate days since earliest credit line
                days_diff = (reference_date - earliest_dates).dt.days

                # Convert to years and months
                df['credit_age_years'] = days_diff / 365.25
                df['credit_age_months'] = days_diff / 30.44

            except Exception as e:
                logger.warning(f"Could not parse earliest_cr_line: {e}")

        # Loan issue date features
        if 'issue_d' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_d'], errors='coerce')

                # Extract components
                df['issue_year'] = issue_dates.dt.year
                df['issue_month'] = issue_dates.dt.month
                df['issue_quarter'] = issue_dates.dt.quarter

                # Season (financial quarters)
                df['loan_season'] = df['issue_month'].map({
                    1: 'Q1', 2: 'Q1', 3: 'Q1',
                    4: 'Q2', 5: 'Q2', 6: 'Q2',
                    7: 'Q3', 8: 'Q3', 9: 'Q3',
                    10: 'Q4', 11: 'Q4', 12: 'Q4'
                })

            except Exception as e:
                logger.warning(f"Could not parse issue_d: {e}")

        # Employment length features
        if 'emp_length' in df.columns:
            # Parse employment length (e.g., "10+ years", "< 1 year")
            df['emp_length_numeric'] = df['emp_length'].replace({
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            })

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")

        # Interest rate × DTI
        if 'int_rate' in df.columns and 'dti' in df.columns:
            df['int_rate_x_dti'] = df['int_rate'] * df['dti']

        # Loan amount × Interest rate
        if 'loan_amnt' in df.columns and 'int_rate' in df.columns:
            df['loan_amnt_x_int_rate'] = df['loan_amnt'] * df['int_rate']

        # Interest rate × Term
        if 'int_rate' in df.columns and 'term' in df.columns:
            # Parse term (e.g., " 36 months", " 60 months")
            term_numeric = df['term'].str.extract('(\d+)', expand=False).astype(float)
            df['int_rate_x_term'] = df['int_rate'] * term_numeric

        # FICO × DTI
        # Check for FICO range columns
        fico_col = None
        if 'fico_range_high' in df.columns:
            fico_col = 'fico_range_high'
        elif 'fico_range_low' in df.columns:
            fico_col = 'fico_range_low'

        if fico_col and 'dti' in df.columns:
            df['fico_x_dti'] = df[fico_col] * df['dti']

        # Income × Employment length
        if 'annual_inc' in df.columns and 'emp_length_numeric' in df.columns:
            df['income_x_emp_length'] = df['annual_inc'] * df['emp_length_numeric']

        return df

    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregation and statistical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with aggregation features added
        """
        logger.info("Creating aggregation features...")

        # Total credit accounts
        total_credit_cols = []
        for col in ['open_acc', 'total_acc', 'num_actv_bc_tl', 'num_bc_tl', 'num_sats']:
            if col in df.columns:
                total_credit_cols.append(col)

        if total_credit_cols:
            df['total_credit_accounts'] = df[total_credit_cols].sum(axis=1)

        # Total balance across all accounts
        balance_cols = []
        for col in ['tot_cur_bal', 'total_bal_ex_mort', 'total_bc_limit', 'total_rev_hi_lim']:
            if col in df.columns:
                balance_cols.append(col)

        if balance_cols:
            df['total_balance_all'] = df[balance_cols].sum(axis=1)

        return df

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned/discretized features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with binned features added
        """
        logger.info("Creating binned features...")

        # Income buckets
        if 'annual_inc' in df.columns:
            if 'income' in self.bin_edges:
                df['income_bucket'] = pd.cut(
                    df['annual_inc'],
                    bins=self.bin_edges['income'],
                    labels=range(len(self.bin_edges['income']) - 1),
                    include_lowest=True
                ).astype(float)
            else:
                # Use fixed bins
                df['income_bucket'] = pd.cut(
                    df['annual_inc'],
                    bins=[0, 30000, 50000, 75000, 100000, np.inf],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)

        # FICO buckets (credit score ranges)
        fico_col = None
        if 'fico_range_high' in df.columns:
            fico_col = 'fico_range_high'
        elif 'fico_range_low' in df.columns:
            fico_col = 'fico_range_low'

        if fico_col:
            df['fico_bucket'] = pd.cut(
                df[fico_col],
                bins=[0, 600, 660, 720, 780, 850],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)

        # Loan amount buckets
        if 'loan_amnt' in df.columns:
            if 'loan_amnt' in self.bin_edges:
                df['loan_amnt_bucket'] = pd.cut(
                    df['loan_amnt'],
                    bins=self.bin_edges['loan_amnt'],
                    labels=range(len(self.bin_edges['loan_amnt']) - 1),
                    include_lowest=True
                ).astype(float)
            else:
                df['loan_amnt_bucket'] = pd.cut(
                    df['loan_amnt'],
                    bins=[0, 5000, 10000, 20000, 30000, np.inf],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)

        # DTI buckets
        if 'dti' in df.columns:
            df['dti_bucket'] = pd.cut(
                df['dti'],
                bins=[0, 10, 20, 30, 40, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)

        return df

    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive risk indicator features for credit default prediction.

        These features are specifically designed to capture default risk patterns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with risk indicator features added
        """
        logger.info("Creating risk indicator features...")

        # 1. Composite Delinquency Risk Score
        delinq_risk = 0
        if 'delinq_2yrs' in df.columns:
            delinq_risk += df['delinq_2yrs'] * 2  # Recent delinquencies weighted higher
        if 'mths_since_last_delinq' in df.columns:
            # Recent delinquencies are riskier
            df['delinq_recency_risk'] = np.where(
                df['mths_since_last_delinq'].isna(),
                0,  # No history
                np.where(df['mths_since_last_delinq'] <= 12, 3,  # Very recent
                np.where(df['mths_since_last_delinq'] <= 24, 2,  # Recent
                np.where(df['mths_since_last_delinq'] <= 36, 1, 0)))  # Old
            )
            delinq_risk += df['delinq_recency_risk']

        if 'pub_rec' in df.columns:
            delinq_risk += df['pub_rec'] * 3  # Public records
        if 'pub_rec_bankruptcies' in df.columns:
            delinq_risk += df['pub_rec_bankruptcies'] * 5  # Bankruptcies very high risk

        df['composite_delinquency_risk'] = delinq_risk

        # 2. Payment Burden Ratio (Monthly payment relative to income)
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_burden_ratio'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
            # High burden flag (>30% of income)
            df['high_payment_burden'] = (df['payment_burden_ratio'] > 0.3).astype(int)

        # 3. Credit Stability Score
        stability_score = 0
        if 'credit_age_years' in df.columns:
            # Longer credit history = more stable
            stability_score += np.clip(df['credit_age_years'] / 10, 0, 3)
        if 'emp_length_numeric' in df.columns:
            # Employment stability
            stability_score += np.clip(df['emp_length_numeric'] / 5, 0, 2)
        if 'home_ownership' in df.columns:
            # Homeownership adds stability
            stability_score += df['home_ownership'].map({
                'OWN': 2, 'MORTGAGE': 1.5, 'RENT': 0.5, 'OTHER': 0, 'NONE': 0
            }).fillna(0)

        df['credit_stability_score'] = stability_score

        # 4. Credit Utilization Risk
        if 'revol_util' in df.columns:
            # Parse revol_util (might be string with %)
            if df['revol_util'].dtype == 'object':
                df['revol_util_numeric'] = pd.to_numeric(
                    df['revol_util'].str.replace('%', ''), errors='coerce'
                )
            else:
                df['revol_util_numeric'] = df['revol_util']

            # High utilization is risky
            df['high_utilization_risk'] = np.where(
                df['revol_util_numeric'] > 80, 3,
                np.where(df['revol_util_numeric'] > 60, 2,
                np.where(df['revol_util_numeric'] > 40, 1, 0))
            )

        # 5. Total Debt Burden
        if 'tot_cur_bal' in df.columns and 'annual_inc' in df.columns:
            df['total_debt_to_income'] = df['tot_cur_bal'] / (df['annual_inc'] + 1)
            df['excessive_debt_flag'] = (df['total_debt_to_income'] > 5).astype(int)

        # 6. Inquiries Risk (Credit shopping behavior)
        if 'inq_last_6mths' in df.columns:
            # Multiple inquiries suggest credit stress
            df['inquiry_risk'] = np.where(
                df['inq_last_6mths'] >= 5, 3,
                np.where(df['inq_last_6mths'] >= 3, 2,
                np.where(df['inq_last_6mths'] >= 1, 1, 0))
            )

        # 7. Account Health Score
        account_health = 0
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            # Ratio of open to total accounts
            df['account_activity_ratio'] = df['open_acc'] / (df['total_acc'] + 1)
            account_health += np.clip(df['account_activity_ratio'] * 3, 0, 3)

        if 'num_actv_bc_tl' in df.columns and 'num_bc_tl' in df.columns:
            # Active bankcard accounts ratio
            df['active_bc_ratio'] = df['num_actv_bc_tl'] / (df['num_bc_tl'] + 1)
            account_health += np.clip(df['active_bc_ratio'] * 2, 0, 2)

        df['account_health_score'] = account_health

        # 8. FICO-DTI Risk Interaction
        if 'fico_range_low' in df.columns and 'dti' in df.columns:
            # Low FICO + High DTI = Very risky
            df['fico_dti_risk'] = (850 - df['fico_range_low']) / 100 * df['dti'] / 10

        # 9. Loan Purpose Risk
        if 'purpose' in df.columns:
            # Some purposes are riskier than others
            high_risk_purposes = ['small_business', 'debt_consolidation', 'credit_card']
            low_risk_purposes = ['home_improvement', 'house', 'car', 'major_purchase']

            df['high_risk_purpose'] = df['purpose'].isin(high_risk_purposes).astype(int)
            df['low_risk_purpose'] = df['purpose'].isin(low_risk_purposes).astype(int)

        # 10. Recent Credit Stress Indicators
        stress_score = 0
        if 'mths_since_recent_bc' in df.columns:
            # Very recent new credit might indicate stress
            stress_score += np.where(df['mths_since_recent_bc'] <= 6, 2, 0)

        if 'mths_since_recent_inq' in df.columns:
            stress_score += np.where(df['mths_since_recent_inq'] <= 3, 2, 0)

        if 'mths_since_recent_revol_delinq' in df.columns:
            stress_score += np.where(df['mths_since_recent_revol_delinq'] <= 12, 3, 0)

        df['credit_stress_score'] = stress_score

        # 11. Income Stability Flag
        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['high_loan_to_income'] = (df['loan_to_income_ratio'] > 0.5).astype(int)

        # 12. Overall Risk Score (weighted combination)
        overall_risk = 0
        if 'composite_delinquency_risk' in df.columns:
            overall_risk += df['composite_delinquency_risk'] * 0.25
        if 'credit_stability_score' in df.columns:
            overall_risk -= df['credit_stability_score'] * 0.15  # Stability reduces risk
        if 'high_utilization_risk' in df.columns:
            overall_risk += df['high_utilization_risk'] * 0.20
        if 'credit_stress_score' in df.columns:
            overall_risk += df['credit_stress_score'] * 0.20
        if 'payment_burden_ratio' in df.columns:
            overall_risk += df['payment_burden_ratio'] * 10 * 0.20

        df['overall_risk_score'] = overall_risk

        # Fill any NaN values created during risk indicator calculation
        # Use 0 for risk scores (no risk if data is missing)
        risk_cols = [
            'composite_delinquency_risk', 'delinq_recency_risk', 'payment_burden_ratio',
            'high_payment_burden', 'credit_stability_score', 'high_utilization_risk',
            'total_debt_to_income', 'excessive_debt_flag', 'inquiry_risk',
            'account_health_score', 'fico_dti_risk', 'high_risk_purpose',
            'low_risk_purpose', 'credit_stress_score', 'loan_to_income_ratio',
            'high_loan_to_income', 'overall_risk_score'
        ]

        for col in risk_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted feature engineer to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted feature engineer")

        save_pickle(self, filepath)
        logger.info(f"Saved feature engineer to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureEngineer':
        """
        Load fitted feature engineer from file.

        Args:
            filepath: Path to saved feature engineer

        Returns:
            Loaded FeatureEngineer instance
        """
        feature_engineer = load_pickle(filepath)
        logger.info(f"Loaded feature engineer from {filepath}")
        return feature_engineer
