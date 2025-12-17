"""
Exposure at Default (EAD) Simulation
Simulates realistic EAD values for defaulted loans based on utilization patterns
EAD = Outstanding balance at the time of default
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'


def simulate_ead(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate EAD values for defaulted loans.

    EAD factors:
    - Loan amount (base exposure)
    - Revolving utilization (how much credit is used)
    - DTI ratio (higher DTI = more likely to max out credit)
    - Time trends (defaults often happen after increasing utilization)

    Args:
        df: DataFrame with loan data

    Returns:
        DataFrame with simulated EAD column
    """
    print("\n" + "="*80)
    print("SIMULATING EXPOSURE AT DEFAULT (EAD)")
    print("="*80)

    # Filter only defaulted loans
    defaulted = df[df['default'] == 1].copy()
    print(f"\nDefaulted loans: {len(defaulted):,}")

    if len(defaulted) == 0:
        print("WARNING: No defaulted loans found!")
        return df

    # Base EAD: Start with 85% of loan amount (typical utilization at default)
    base_utilization = 0.85

    print("\nCalculating EAD based on loan characteristics...")

    # Initialize EAD as percentage of loan amount
    utilization = np.full(len(defaulted), base_utilization)

    # 1. Revolving Utilization Impact
    # Higher revolving utilization = tendency to use more credit = higher EAD
    if 'revol_util' in defaulted.columns:
        revol_util_norm = defaulted['revol_util'].fillna(50) / 100  # Convert to 0-1
        revol_impact = 0.10 * revol_util_norm  # Up to +10% utilization
        utilization += revol_impact
        print(f"  ✓ Revolving utilization adjustment: {revol_impact.mean():.4f} avg")

    # 2. DTI Ratio Impact
    # Higher DTI = more financial stress = higher utilization at default
    if 'dti' in defaulted.columns:
        dti_norm = np.clip(defaulted['dti'] / 50, 0, 1)  # Normalize, cap at 50
        dti_impact = 0.08 * dti_norm  # Up to +8% utilization
        utilization += dti_impact
        print(f"  ✓ DTI ratio adjustment: {dti_impact.mean():.4f} avg")

    # 3. Credit Score Impact
    # Lower credit score = worse financial management = higher utilization
    if 'fico_range_low' in defaulted.columns:
        fico_norm = (defaulted['fico_range_low'] - 600) / (850 - 600)  # Normalize to 0-1
        fico_impact = -0.06 * fico_norm  # Up to -6% utilization for high FICO
        utilization += fico_impact
        print(f"  ✓ Credit score adjustment: {fico_impact.mean():.4f} avg")

    # 4. Delinquency History Impact
    # Recent delinquencies = financial stress = higher utilization
    if 'delinq_2yrs' in defaulted.columns:
        delinq_impact = np.minimum(defaulted['delinq_2yrs'] * 0.02, 0.05)  # Up to +5%
        utilization += delinq_impact
        print(f"  ✓ Delinquency adjustment: {delinq_impact.mean():.4f} avg")

    # 5. Add random noise (market conditions, timing, etc.)
    noise = np.random.normal(0, 0.05, len(defaulted))  # 5% std deviation
    utilization += noise
    print(f"  ✓ Random noise added: ±{noise.std():.4f} std")

    # Clip to valid range [0.5, 1.0] - defaults rarely happen with low utilization
    utilization = np.clip(utilization, 0.50, 1.00)

    # Calculate EAD as utilization * loan amount
    ead = utilization * defaulted['loan_amnt']

    # Assign EAD to defaulted loans
    defaulted['ead'] = ead
    defaulted['ead_utilization'] = utilization

    # For non-defaulted loans, set EAD to NaN (not applicable)
    df = df.copy()
    df['ead'] = np.nan
    df['ead_utilization'] = np.nan
    df.loc[df['default'] == 1, 'ead'] = ead
    df.loc[df['default'] == 1, 'ead_utilization'] = utilization

    print("\n" + "="*80)
    print("EAD SIMULATION COMPLETE")
    print("="*80)
    print(f"\nEAD Statistics (for defaulted loans only):")
    print(f"  Mean EAD Amount:      £{ead.mean():,.2f}")
    print(f"  Median EAD Amount:    £{np.median(ead):,.2f}")
    print(f"  Mean Utilization:     {utilization.mean():.2%}")
    print(f"  Median Utilization:   {np.median(utilization):.2%}")
    print(f"  Min Utilization:      {utilization.min():.2%}")
    print(f"  Max Utilization:      {utilization.max():.2%}")

    return df


def main():
    """Main execution: Load data, simulate EAD, save results."""
    print("\n" + "="*80)
    print("PHASE 4: EAD DATA SIMULATION")
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

    # Simulate EAD
    print("\n2. Simulating EAD values...")
    train_data = simulate_ead(train_data)

    # Save EAD data
    print("\n3. Saving EAD data...")
    ead_dir = DATA_DIR / 'ead'
    ead_dir.mkdir(exist_ok=True)

    # Save full data with EAD
    ead_path = ead_dir / 'train_with_ead.csv'
    train_data.to_csv(ead_path, index=False)
    print(f"✓ Saved to: {ead_path}")

    # Save only defaulted loans with EAD (for modeling)
    defaulted_ead = train_data[train_data['default'] == 1].copy()
    defaulted_path = ead_dir / 'defaulted_loans_ead.csv'
    defaulted_ead.to_csv(defaulted_path, index=False)
    print(f"✓ Saved defaulted loans to: {defaulted_path}")

    print("\n" + "="*80)
    print("✅ EAD SIMULATION COMPLETE!")
    print("="*80)
    print(f"\nNext step: Train EAD regression models using {len(defaulted_ead):,} defaulted loans")


if __name__ == '__main__':
    main()
