"""
Stress Testing Framework for Credit Risk Models

This module implements three economic scenarios for stress testing:
1. Baseline: Normal economic conditions
2. Adverse: Moderate economic stress
3. Severe: Extreme economic stress

Since the dataset lacks explicit macroeconomic variables, we simulate
their impact by adjusting borrower characteristics that correlate with
economic conditions.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EconomicScenario:
    """Defines macroeconomic scenarios for stress testing."""

    def __init__(self, name, description, adjustments):
        """
        Args:
            name: Scenario name (e.g., "Baseline", "Adverse", "Severe")
            description: Description of economic conditions
            adjustments: Dictionary of feature adjustments
        """
        self.name = name
        self.description = description
        self.adjustments = adjustments

    def __repr__(self):
        return f"EconomicScenario(name='{self.name}')"


# Define three economic scenarios
SCENARIOS = {
    'baseline': EconomicScenario(
        name='Baseline',
        description='Normal economic conditions - No stress',
        adjustments={
            # No adjustments - use original values
            'dti_multiplier': 1.00,              # Debt-to-Income stays same
            'revol_util_increase': 0.00,         # Revolving utilization unchanged
            'income_multiplier': 1.00,           # Income unchanged
            'delinq_increase': 0.00,             # No increase in delinquencies
            'fico_decrease': 0,                  # FICO scores unchanged
            'unemployment_proxy': 0.00,          # No job loss impact
        }
    ),

    'adverse': EconomicScenario(
        name='Adverse',
        description='Moderate economic stress - Recession scenario',
        adjustments={
            # Moderate deterioration in borrower conditions
            'dti_multiplier': 1.15,              # DTI increases by 15%
            'revol_util_increase': 0.10,         # Credit utilization +10pp
            'income_multiplier': 0.95,           # Income decreases by 5%
            'delinq_increase': 0.30,             # 30% of borrowers add delinquency
            'fico_decrease': 20,                 # FICO drops by 20 points
            'unemployment_proxy': 0.05,          # 5% employment impact
        }
    ),

    'severe': EconomicScenario(
        name='Severe',
        description='Extreme economic stress - Financial crisis scenario',
        adjustments={
            # Severe deterioration in borrower conditions
            'dti_multiplier': 1.35,              # DTI increases by 35%
            'revol_util_increase': 0.20,         # Credit utilization +20pp
            'income_multiplier': 0.85,           # Income decreases by 15%
            'delinq_increase': 0.50,             # 50% of borrowers add delinquency
            'fico_decrease': 40,                 # FICO drops by 40 points
            'unemployment_proxy': 0.10,          # 10% employment impact
        }
    )
}


class StressTestingEngine:
    """Engine for running stress tests on credit risk models."""

    def __init__(self):
        """Initialize the stress testing engine."""
        self.pd_models = {}
        self.pd_weights = None
        self.pd_model_names = None
        self.feature_columns = None
        self.original_data = None
        self.scenario_results = {}

    def load_pd_ensemble(self):
        """Load the champion PD ensemble model."""
        print("\n" + "="*80)
        print("LOADING CHAMPION PD MODEL (WEIGHTED ENSEMBLE)")
        print("="*80)

        # Model names and weights from the champion ensemble
        self.pd_model_names = [
            'logistic_regression',
            'random_forest',
            'xgboost',
            'neural_network'
        ]

        # Load weights from stacking ensemble metrics
        weights_path = RESULTS_DIR / 'weighted_ensemble_metrics.pkl'
        if weights_path.exists():
            with open(weights_path, 'rb') as f:
                data = pickle.load(f)
                self.pd_weights = data.get('weights', np.array([0.25, 0.25, 0.25, 0.25]))
        else:
            # Default equal weights
            self.pd_weights = np.array([0.25, 0.25, 0.25, 0.25])

        print(f"\nModel weights: {dict(zip(self.pd_model_names, self.pd_weights))}")

        # Load individual models
        models_loaded = []
        for name in self.pd_model_names:
            try:
                model_path = MODELS_DIR / f'{name}_model.pkl'
                if name == 'neural_network':
                    model_path = MODELS_DIR / f'{name}_model.keras'

                if not model_path.exists():
                    print(f"  Warning: {name} model not found")
                    continue

                if name == 'neural_network':
                    from tensorflow import keras
                    self.pd_models[name] = keras.models.load_model(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        self.pd_models[name] = pickle.load(f)

                models_loaded.append(name)
                print(f"  [OK] Loaded {name}")

            except Exception as e:
                print(f"  Warning: Could not load {name}: {str(e)[:60]}")
                continue

        # Adjust weights for loaded models only
        if len(models_loaded) < len(self.pd_model_names):
            loaded_indices = [i for i, name in enumerate(self.pd_model_names)
                            if name in models_loaded]
            self.pd_weights = self.pd_weights[loaded_indices]
            self.pd_weights = self.pd_weights / self.pd_weights.sum()
            self.pd_model_names = models_loaded

        print(f"\n[OK] Loaded {len(self.pd_models)} models for stress testing")

    def load_test_data(self):
        """Load test data for stress testing."""
        print("\n" + "="*80)
        print("LOADING TEST DATA")
        print("="*80)

        test_path = DATA_DIR / 'test.csv'
        self.original_data = pd.read_csv(test_path)

        # Remove target column if present
        if 'default' in self.original_data.columns:
            print(f"\nLoaded {len(self.original_data):,} samples")
            print("(Removed 'default' column for prediction)")
            self.original_data = self.original_data.drop('default', axis=1)
        else:
            print(f"\nLoaded {len(self.original_data):,} samples")

        self.feature_columns = self.original_data.columns.tolist()
        print(f"Features: {len(self.feature_columns)}")

    def apply_scenario(self, scenario):
        """
        Apply economic scenario adjustments to data.

        Args:
            scenario: EconomicScenario object

        Returns:
            DataFrame with adjusted features
        """
        print(f"\n  Applying {scenario.name} scenario adjustments...")

        # Create a copy of original data
        stressed_data = self.original_data.copy()
        adj = scenario.adjustments

        # 1. Adjust DTI (Debt-to-Income)
        if 'dti' in stressed_data.columns:
            stressed_data['dti'] = stressed_data['dti'] * adj['dti_multiplier']
            print(f"    - DTI adjusted by {(adj['dti_multiplier']-1)*100:+.0f}%")

        # 2. Adjust Revolving Utilization
        if 'revol_util' in stressed_data.columns:
            stressed_data['revol_util'] = np.minimum(
                stressed_data['revol_util'] + (adj['revol_util_increase'] * 100),
                100.0  # Cap at 100%
            )
            print(f"    - Revolving utilization +{adj['revol_util_increase']*100:.0f} percentage points")

        # 3. Adjust Income (inverse effect on DTI and other income ratios)
        if 'annual_inc' in stressed_data.columns:
            stressed_data['annual_inc'] = stressed_data['annual_inc'] * adj['income_multiplier']
            print(f"    - Annual income adjusted by {(adj['income_multiplier']-1)*100:+.0f}%")

        # 4. Adjust Delinquencies
        if 'delinq_2yrs' in stressed_data.columns:
            # Add delinquencies to a percentage of borrowers
            random_mask = np.random.random(len(stressed_data)) < adj['delinq_increase']
            stressed_data.loc[random_mask, 'delinq_2yrs'] = (
                stressed_data.loc[random_mask, 'delinq_2yrs'] + 1
            )
            print(f"    - {adj['delinq_increase']*100:.0f}% of borrowers add 1 delinquency")

        # 5. Adjust FICO scores
        if 'fico_range_low' in stressed_data.columns:
            stressed_data['fico_range_low'] = np.maximum(
                stressed_data['fico_range_low'] - adj['fico_decrease'],
                300  # FICO minimum
            )
            print(f"    - FICO scores decreased by {adj['fico_decrease']} points")

        if 'fico_range_high' in stressed_data.columns:
            stressed_data['fico_range_high'] = np.maximum(
                stressed_data['fico_range_high'] - adj['fico_decrease'],
                300
            )

        # 6. Simulate unemployment impact (increase in recent inquiries)
        if 'inq_last_6mths' in stressed_data.columns:
            random_mask = np.random.random(len(stressed_data)) < adj['unemployment_proxy']
            stressed_data.loc[random_mask, 'inq_last_6mths'] = (
                stressed_data.loc[random_mask, 'inq_last_6mths'] + 2  # Add 2 inquiries
            )
            print(f"    - {adj['unemployment_proxy']*100:.0f}% of borrowers add 2 credit inquiries")

        # Recalculate derived features if they exist
        if 'loan_to_income' in stressed_data.columns and 'loan_amnt' in stressed_data.columns:
            stressed_data['loan_to_income'] = (
                stressed_data['loan_amnt'] / stressed_data['annual_inc']
            )

        return stressed_data

    def predict_pd(self, X):
        """
        Predict PD using the ensemble model.

        Args:
            X: DataFrame with features

        Returns:
            Array of PD predictions
        """
        predictions = []

        for name in self.pd_model_names:
            model = self.pd_models[name]

            if name == 'neural_network':
                # Neural network needs scaling - load scaler
                scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X_scaled = scaler.transform(X)
                    pred_proba = model.predict(X_scaled, verbose=0)
                    predictions.append(pred_proba.flatten())
                else:
                    # Fallback without scaling
                    pred_proba = model.predict(X.values, verbose=0)
                    predictions.append(pred_proba.flatten())
            else:
                pred_proba = model.predict_proba(X)[:, 1]
                predictions.append(pred_proba)

        # Weighted average
        predictions = np.array(predictions)
        pd_probs = np.average(predictions, axis=0, weights=self.pd_weights)

        return pd_probs

    def run_scenario(self, scenario_name):
        """
        Run stress test for a specific scenario.

        Args:
            scenario_name: 'baseline', 'adverse', or 'severe'

        Returns:
            DataFrame with scenario results
        """
        scenario = SCENARIOS[scenario_name]

        print("\n" + "="*80)
        print(f"RUNNING {scenario.name.upper()} SCENARIO")
        print("="*80)
        print(f"\nDescription: {scenario.description}")

        # Apply scenario adjustments
        stressed_data = self.apply_scenario(scenario)

        # Predict PD
        print(f"\n  Predicting PD for {scenario.name} scenario...")
        pd_probs = self.predict_pd(stressed_data)

        # Create results DataFrame
        results = pd.DataFrame({
            'Scenario': scenario.name,
            'PD': pd_probs
        })

        # Store results
        self.scenario_results[scenario_name] = {
            'scenario': scenario,
            'data': stressed_data,
            'results': results,
            'pd_probs': pd_probs
        }

        # Print statistics
        print(f"\n  {scenario.name} PD Statistics:")
        print(f"    Mean PD:   {pd_probs.mean():.2%}")
        print(f"    Median PD: {np.median(pd_probs):.2%}")
        print(f"    Min PD:    {pd_probs.min():.2%}")
        print(f"    Max PD:    {pd_probs.max():.2%}")

        return results

    def run_all_scenarios(self):
        """Run all three scenarios and compare results."""
        print("\n" + "="*80)
        print("STRESS TESTING: RUNNING ALL SCENARIOS")
        print("="*80)

        # Run each scenario
        for scenario_name in ['baseline', 'adverse', 'severe']:
            self.run_scenario(scenario_name)

    def compare_scenarios(self):
        """Compare results across all scenarios."""
        print("\n" + "="*80)
        print("SCENARIO COMPARISON")
        print("="*80)

        # Collect PD statistics for each scenario
        comparison_data = []

        for scenario_name in ['baseline', 'adverse', 'severe']:
            if scenario_name in self.scenario_results:
                pd_probs = self.scenario_results[scenario_name]['pd_probs']

                comparison_data.append({
                    'Scenario': SCENARIOS[scenario_name].name,
                    'Mean_PD': pd_probs.mean(),
                    'Median_PD': np.median(pd_probs),
                    'Std_PD': pd_probs.std(),
                    'P10': np.percentile(pd_probs, 10),
                    'P25': np.percentile(pd_probs, 25),
                    'P75': np.percentile(pd_probs, 75),
                    'P90': np.percentile(pd_probs, 90),
                    'Max_PD': pd_probs.max()
                })

        comparison_df = pd.DataFrame(comparison_data)

        print("\nPD Statistics Across Scenarios:\n")
        print(comparison_df.to_string(index=False))

        # Calculate stress impact
        print("\n" + "="*80)
        print("STRESS IMPACT ANALYSIS")
        print("="*80)

        if 'baseline' in self.scenario_results:
            baseline_pd = self.scenario_results['baseline']['pd_probs'].mean()

            print(f"\nBaseline Mean PD: {baseline_pd:.2%}")

            for scenario_name in ['adverse', 'severe']:
                if scenario_name in self.scenario_results:
                    scenario_pd = self.scenario_results[scenario_name]['pd_probs'].mean()
                    increase = scenario_pd - baseline_pd
                    pct_increase = (increase / baseline_pd) * 100

                    print(f"\n{SCENARIOS[scenario_name].name} Scenario:")
                    print(f"  Mean PD: {scenario_pd:.2%}")
                    print(f"  Absolute increase: +{increase:.2%}")
                    print(f"  Relative increase: +{pct_increase:.1f}%")

        # Save comparison results
        output_path = RESULTS_DIR / 'stress_testing_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\n[OK] Comparison saved to: {output_path}")

        return comparison_df

    def save_detailed_results(self):
        """Save detailed results for each scenario."""
        print("\n" + "="*80)
        print("SAVING DETAILED RESULTS")
        print("="*80)

        for scenario_name in ['baseline', 'adverse', 'severe']:
            if scenario_name in self.scenario_results:
                results = self.scenario_results[scenario_name]['results']

                output_path = RESULTS_DIR / f'stress_testing_{scenario_name}_pd.csv'
                results.to_csv(output_path, index=False)
                print(f"  [OK] {scenario_name.capitalize()} scenario saved to: {output_path.name}")


def main():
    """Main execution: Run stress testing on all scenarios."""
    print("\n" + "="*80)
    print("CREDIT RISK STRESS TESTING FRAMEWORK")
    print("="*80)
    print("\nPhase 5: Stress Testing Scenario Models")
    print("Re-scoring Champion PD Model under different economic scenarios\n")

    # Initialize engine
    engine = StressTestingEngine()

    # Load champion PD model
    engine.load_pd_ensemble()

    # Load test data
    engine.load_test_data()

    # Run all scenarios
    engine.run_all_scenarios()

    # Compare scenarios
    engine.compare_scenarios()

    # Save detailed results
    engine.save_detailed_results()

    print("\n" + "="*80)
    print("[COMPLETE] STRESS TESTING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
