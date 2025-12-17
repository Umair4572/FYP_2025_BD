"""
Comprehensive Stress Testing with Expected Loss Calculation

This module extends stress testing to calculate Expected Loss (PD × LGD × EAD)
under different economic scenarios, providing a complete view of portfolio risk.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import stress_testing
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the stress testing engine
from stress_testing import StressTestingEngine, SCENARIOS

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'


class StressTestingWithEL(StressTestingEngine):
    """Extended stress testing engine that calculates Expected Loss."""

    def __init__(self):
        """Initialize the extended stress testing engine."""
        super().__init__()
        self.lgd_rf_model = None
        self.lgd_xgb_model = None
        self.ead_rf_model = None
        self.ead_xgb_model = None
        self.lgd_features = None
        self.ead_features = None

    def load_lgd_ead_models(self):
        """Load LGD and EAD models for Expected Loss calculation."""
        print("\n" + "="*80)
        print("LOADING LGD AND EAD MODELS")
        print("="*80)

        # Load LGD models
        try:
            lgd_rf_path = MODELS_DIR / 'lgd_random_forest.pkl'
            lgd_xgb_path = MODELS_DIR / 'lgd_xgboost.pkl'

            with open(lgd_rf_path, 'rb') as f:
                self.lgd_rf_model = pickle.load(f)

            with open(lgd_xgb_path, 'rb') as f:
                self.lgd_xgb_model = pickle.load(f)

            # Load feature columns from metrics
            lgd_metrics_path = RESULTS_DIR / 'lgd_metrics.pkl'
            with open(lgd_metrics_path, 'rb') as f:
                lgd_data = pickle.load(f)
            self.lgd_features = lgd_data['feature_columns']

            print("  [OK] LGD Models loaded")

        except Exception as e:
            print(f"  [FAIL] LGD Models loading failed: {str(e)}")

        # Load EAD models
        try:
            ead_rf_path = MODELS_DIR / 'ead_random_forest.pkl'
            ead_xgb_path = MODELS_DIR / 'ead_xgboost.pkl'

            with open(ead_rf_path, 'rb') as f:
                self.ead_rf_model = pickle.load(f)

            with open(ead_xgb_path, 'rb') as f:
                self.ead_xgb_model = pickle.load(f)

            # Load feature columns from metrics
            ead_metrics_path = RESULTS_DIR / 'ead_metrics.pkl'
            with open(ead_metrics_path, 'rb') as f:
                ead_data = pickle.load(f)
            self.ead_features = ead_data['feature_columns']

            print("  [OK] EAD Models loaded")

        except Exception as e:
            print(f"  [FAIL] EAD Models loading failed: {str(e)}")

    def predict_lgd(self, X):
        """Predict Loss Given Default."""
        X_lgd = X[self.lgd_features]

        lgd_rf = self.lgd_rf_model.predict(X_lgd)
        lgd_xgb = self.lgd_xgb_model.predict(X_lgd)

        # Ensemble (average)
        lgd = (lgd_rf + lgd_xgb) / 2
        lgd = np.clip(lgd, 0, 1)

        return lgd

    def predict_ead(self, X):
        """Predict Exposure at Default."""
        X_ead = X[self.ead_features]

        ead_rf = self.ead_rf_model.predict(X_ead)
        ead_xgb = self.ead_xgb_model.predict(X_ead)

        # Ensemble (average) - this gives us standardized EAD
        ead_standardized = (ead_rf + ead_xgb) / 2

        # Reverse standardization to get actual pound amounts
        loan_amnt_mean = 14262.275
        loan_amnt_std = 8379.608

        ead_actual = (ead_standardized * loan_amnt_std) + loan_amnt_mean
        ead_actual = np.clip(ead_actual, 0, 35000)

        return ead_actual

    def run_scenario_with_el(self, scenario_name):
        """
        Run stress test for a specific scenario with Expected Loss.

        Args:
            scenario_name: 'baseline', 'adverse', or 'severe'

        Returns:
            DataFrame with complete scenario results
        """
        scenario = SCENARIOS[scenario_name]

        print("\n" + "="*80)
        print(f"RUNNING {scenario.name.upper()} SCENARIO (WITH EXPECTED LOSS)")
        print("="*80)
        print(f"\nDescription: {scenario.description}")

        # Apply scenario adjustments
        stressed_data = self.apply_scenario(scenario)

        # Predict PD
        print(f"\n  Predicting PD for {scenario.name} scenario...")
        pd_probs = self.predict_pd(stressed_data)

        # Predict LGD (also affected by economic conditions)
        print(f"  Predicting LGD for {scenario.name} scenario...")
        lgd_values = self.predict_lgd(stressed_data)

        # In adverse/severe scenarios, LGD typically increases
        if scenario_name == 'adverse':
            lgd_values = np.minimum(lgd_values * 1.10, 1.0)  # +10% increase
            print(f"    (LGD increased by 10% due to economic stress)")
        elif scenario_name == 'severe':
            lgd_values = np.minimum(lgd_values * 1.20, 1.0)  # +20% increase
            print(f"    (LGD increased by 20% due to severe economic stress)")

        # Predict EAD
        print(f"  Predicting EAD for {scenario.name} scenario...")
        ead_values = self.predict_ead(stressed_data)

        # Calculate Expected Loss
        print(f"  Calculating Expected Loss...")
        expected_loss = pd_probs * lgd_values * ead_values

        # Create results DataFrame
        results = pd.DataFrame({
            'Scenario': scenario.name,
            'PD': pd_probs,
            'LGD': lgd_values,
            'EAD': ead_values,
            'Expected_Loss': expected_loss
        })

        # Store results
        self.scenario_results[scenario_name] = {
            'scenario': scenario,
            'data': stressed_data,
            'results': results,
            'pd_probs': pd_probs,
            'lgd_values': lgd_values,
            'ead_values': ead_values,
            'expected_loss': expected_loss
        }

        # Print statistics
        print(f"\n  {scenario.name} Statistics:")
        print(f"    Mean PD:            {pd_probs.mean():.2%}")
        print(f"    Mean LGD:           {lgd_values.mean():.2%}")
        print(f"    Mean EAD:           £{ead_values.mean():,.2f}")
        print(f"    Mean Expected Loss: £{expected_loss.mean():,.2f}")
        print(f"    Total Expected Loss: £{expected_loss.sum():,.2f}")

        return results

    def run_all_scenarios_with_el(self):
        """Run all three scenarios with Expected Loss calculation."""
        print("\n" + "="*80)
        print("STRESS TESTING: RUNNING ALL SCENARIOS WITH EXPECTED LOSS")
        print("="*80)

        # Run each scenario
        for scenario_name in ['baseline', 'adverse', 'severe']:
            self.run_scenario_with_el(scenario_name)

    def compare_scenarios_with_el(self):
        """Compare Expected Loss across all scenarios."""
        print("\n" + "="*80)
        print("EXPECTED LOSS COMPARISON ACROSS SCENARIOS")
        print("="*80)

        # Collect statistics for each scenario
        comparison_data = []

        for scenario_name in ['baseline', 'adverse', 'severe']:
            if scenario_name in self.scenario_results:
                res = self.scenario_results[scenario_name]
                pd_probs = res['pd_probs']
                lgd_values = res['lgd_values']
                ead_values = res['ead_values']
                expected_loss = res['expected_loss']

                comparison_data.append({
                    'Scenario': SCENARIOS[scenario_name].name,
                    'Mean_PD': pd_probs.mean(),
                    'Mean_LGD': lgd_values.mean(),
                    'Mean_EAD': ead_values.mean(),
                    'Mean_EL': expected_loss.mean(),
                    'Median_EL': np.median(expected_loss),
                    'Total_EL': expected_loss.sum(),
                    'P75_EL': np.percentile(expected_loss, 75),
                    'P90_EL': np.percentile(expected_loss, 90),
                    'Max_EL': expected_loss.max()
                })

        comparison_df = pd.DataFrame(comparison_data)

        print("\nExpected Loss Across Scenarios:\n")
        print(comparison_df.to_string(index=False))

        # Calculate stress impact
        print("\n" + "="*80)
        print("STRESS IMPACT ON PORTFOLIO EXPECTED LOSS")
        print("="*80)

        if 'baseline' in self.scenario_results:
            baseline_el = self.scenario_results['baseline']['expected_loss'].sum()

            print(f"\nBaseline Total Expected Loss: £{baseline_el:,.2f}")

            for scenario_name in ['adverse', 'severe']:
                if scenario_name in self.scenario_results:
                    scenario_el = self.scenario_results[scenario_name]['expected_loss'].sum()
                    increase = scenario_el - baseline_el
                    pct_increase = (increase / baseline_el) * 100

                    print(f"\n{SCENARIOS[scenario_name].name} Scenario:")
                    print(f"  Total Expected Loss: £{scenario_el:,.2f}")
                    print(f"  Absolute increase: £{increase:,.2f}")
                    print(f"  Relative increase: +{pct_increase:.1f}%")

        # Save comparison results
        output_path = RESULTS_DIR / 'stress_testing_el_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\n[OK] Comparison saved to: {output_path}")

        return comparison_df

    def save_detailed_results_with_el(self):
        """Save detailed results with Expected Loss for each scenario."""
        print("\n" + "="*80)
        print("SAVING DETAILED RESULTS WITH EXPECTED LOSS")
        print("="*80)

        for scenario_name in ['baseline', 'adverse', 'severe']:
            if scenario_name in self.scenario_results:
                results = self.scenario_results[scenario_name]['results']

                output_path = RESULTS_DIR / f'stress_testing_{scenario_name}_el.csv'
                results.to_csv(output_path, index=False)
                print(f"  [OK] {scenario_name.capitalize()} saved: {output_path.name}")


def main():
    """Main execution: Run comprehensive stress testing with Expected Loss."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STRESS TESTING WITH EXPECTED LOSS")
    print("="*80)
    print("\nPhase 5: Stress Testing Scenario Models")
    print("Re-scoring PD, LGD, EAD under different economic scenarios\n")

    # Initialize extended engine
    engine = StressTestingWithEL()

    # Load all models
    engine.load_pd_ensemble()
    engine.load_lgd_ead_models()

    # Load test data
    engine.load_test_data()

    # Run all scenarios with Expected Loss
    engine.run_all_scenarios_with_el()

    # Compare scenarios
    engine.compare_scenarios_with_el()

    # Save detailed results
    engine.save_detailed_results_with_el()

    print("\n" + "="*80)
    print("[COMPLETE] COMPREHENSIVE STRESS TESTING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
