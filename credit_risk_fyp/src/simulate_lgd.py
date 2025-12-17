"""
Loss Given Default (LGD) Simulation
Simulates realistic LGD values for defaulted loans based on loan characteristics
LGD = (Loan Amount - Recovery Amount) / Loan Amount
Range: 0% (full recovery) to 100% (total loss)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'


def simulate_lgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate LGD values for defaulted loans.

    LGD factors:
    - Credit score (higher score = better recovery = lower LGD)
    - Loan amount (higher amount = harder to recover = higher LGD)
    - Home ownership (homeowners = collateral = lower LGD)
    - DTI ratio (higher DTI = worse recovery = higher LGD)
    - Interest rate (higher rate = riskier loan = higher LGD)

    Args:
        df: DataFrame with loan data

    Returns:
        DataFrame with simulated LGD column
    """
    print("\n" + "="*80)
    print("SIMULATING LOSS GIVEN DEFAULT (LGD)")
    print("="*80)

    # Filter only defaulted loans
    defaulted = df[df['default'] == 1].copy()
    print(f"\nDefaulted loans: {len(defaulted):,}")

    if len(defaulted) == 0:
        print("WARNING: No defaulted loans found!")
        return df

    # Base LGD: Start with 45% (industry average for unsecured loans)
    base_lgd = 0.45

    # Initialize LGD
    lgd = np.full(len(defaulted), base_lgd)

    print("\nApplying LGD adjustments based on loan characteristics...")

    # 1. Credit Score Impact (fico_range_low)
    # Higher credit score = better recovery = lower LGD
    if 'fico_range_low' in defaulted.columns:
        fico_norm = (defaulted['fico_range_low'] - 600) / (850 - 600)  # Normalize to 0-1
        fico_impact = -0.15 * fico_norm  # Up to -15% LGD for high FICO
        lgd += fico_impact
        print(f"  ✓ Credit score adjustment: {fico_impact.mean():.4f} avg")

    # 2. Loan Amount Impact
    # Higher loan amount = harder to recover = higher LGD
    if 'loan_amnt' in defaulted.columns:
        loan_norm = np.log1p(defaulted['loan_amnt']) / np.log1p(defaulted['loan_amnt'].max())
        loan_impact = 0.10 * loan_norm  # Up to +10% LGD for large loans
        lgd += loan_impact
        print(f"  ✓ Loan amount adjustment: {loan_impact.mean():.4f} avg")

    # 3. Home Ownership Impact
    # Homeowners have collateral = better recovery = lower LGD
    if 'home_ownership' in defaulted.columns:
        home_impact = np.where(
            defaulted['home_ownership'].isin(['OWN', 'MORTGAGE']),
            -0.10,  # -10% LGD for homeowners
            0.05    # +5% LGD for renters/others
        )
        lgd += home_impact
        print(f"  ✓ Home ownership adjustment: {home_impact.mean():.4f} avg")

    # 4. DTI Ratio Impact
    # Higher DTI = worse financial situation = higher LGD
    if 'dti' in defaulted.columns:
        dti_norm = np.clip(defaulted['dti'] / 50, 0, 1)  # Normalize, cap at 50
        dti_impact = 0.08 * dti_norm  # Up to +8% LGD for high DTI
        lgd += dti_impact
        print(f"  ✓ DTI ratio adjustment: {dti_impact.mean():.4f} avg")

    # 5. Annual Income Impact
    # Higher income = better recovery ability = lower LGD
    if 'annual_inc' in defaulted.columns:
        income_norm = np.log1p(defaulted['annual_inc']) / np.log1p(defaulted['annual_inc'].max())
        income_impact = -0.08 * income_norm  # Up to -8% LGD for high income
        lgd += income_impact
        print(f"  ✓ Annual income adjustment: {income_impact.mean():.4f} avg")

    # 6. Add random noise (represents market conditions, legal costs, etc.)
    noise = np.random.normal(0, 0.05, len(defaulted))  # 5% std deviation
    lgd += noise
    print(f"  ✓ Random noise added: ±{noise.std():.4f} std")

    # Clip to valid range [0, 1]
    lgd = np.clip(lgd, 0.05, 0.95)  # Keep between 5% and 95%

    # Assign LGD to defaulted loans
    defaulted['lgd'] = lgd

    # For non-defaulted loans, set LGD to NaN (not applicable)
    df = df.copy()
    df['lgd'] = np.nan
    df.loc[df['default'] == 1, 'lgd'] = lgd

    print("\n" + "="*80)
    print("LGD SIMULATION COMPLETE")
    print("="*80)
    print(f"\nLGD Statistics (for defaulted loans only):")
    print(f"  Mean LGD:   {lgd.mean():.2%}")
    print(f"  Median LGD: {np.median(lgd):.2%}")
    print(f"  Min LGD:    {lgd.min():.2%}")
    print(f"  Max LGD:    {lgd.max():.2%}")
    print(f"  Std Dev:    {lgd.std():.2%}")

    return df


def main():
    """Main execution: Load data, simulate LGD, save results."""
    print("\n" + "="*80)
    print("PHASE 3: LGD DATA SIMULATION")
    print("="*80)

    # Load processed training data
    print("\n1. Loading processed data...")
    train_path = PROCESSED_DIR / 'train_smote.csv'

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Please run the data pipeline first!")
        return

    train_data = pd.read_csv(train_path)

    print(f"Training data: {len(train_data):,} samples")
    print(f"Defaulted loans: {(train_data['default'] == 1).sum():,}")

    # Simulate LGD
    print("\n2. Simulating LGD values...")
    train_data = simulate_lgd(train_data)

    # Save LGD data
    print("\n3. Saving LGD data...")
    lgd_dir = DATA_DIR / 'lgd'
    lgd_dir.mkdir(exist_ok=True)

    # Save full data with LGD
    lgd_path = lgd_dir / 'train_with_lgd.csv'
    train_data.to_csv(lgd_path, index=False)
    print(f"✓ Saved to: {lgd_path}")

    # Save only defaulted loans with LGD (for modeling)
    defaulted_lgd = train_data[train_data['default'] == 1].copy()
    defaulted_path = lgd_dir / 'defaulted_loans_lgd.csv'
    defaulted_lgd.to_csv(defaulted_path, index=False)
    print(f"✓ Saved defaulted loans to: {defaulted_path}")

    print("\n" + "="*80)
    print("✅ LGD SIMULATION COMPLETE!")
    print("="*80)
    print(f"\nNext step: Train LGD regression models using {len(defaulted_lgd):,} defaulted loans")


if __name__ == '__main__':
    main()
