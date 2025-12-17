"""
Counterfactual Recourse Engine for Credit Risk Assessment

This module generates actionable "what-if" scenarios for rejected loan applicants,
showing the minimal changes needed to achieve loan approval. The engine uses the
Champion PD model with feasibility constraints to ensure recommendations are
realistic and achievable.

Inspired by DiCE (Diverse Counterfactual Explanations) methodology.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'credit_risk_fyp' / 'results'

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureConstraints:
    """Defines feature mutability and feasibility constraints."""

    def __init__(self):
        """Initialize feature constraints."""
        # Immutable features (cannot be changed)
        self.immutable_features = [
            'id',
            'earliest_cr_line',  # Credit history start date
            'zip_code',
            'addr_state',
            'application_type',
            'initial_list_status',
        ]

        # Actionable features (can be changed by applicant)
        self.actionable_features = {
            # Credit behavior (improvable)
            'fico_range_low': {'direction': 'increase', 'max_change': 100, 'min_value': 300, 'max_value': 850},
            'fico_range_high': {'direction': 'increase', 'max_change': 100, 'min_value': 300, 'max_value': 850},
            'revol_util': {'direction': 'decrease', 'max_change': 50, 'min_value': 0, 'max_value': 100},
            'dti': {'direction': 'decrease', 'max_change': 20, 'min_value': 0, 'max_value': 100},
            'delinq_2yrs': {'direction': 'decrease', 'max_change': 10, 'min_value': 0, 'max_value': None},
            'inq_last_6mths': {'direction': 'decrease', 'max_change': 10, 'min_value': 0, 'max_value': None},
            'pub_rec': {'direction': 'decrease', 'max_change': 5, 'min_value': 0, 'max_value': None},
            'acc_now_delinq': {'direction': 'decrease', 'max_change': 5, 'min_value': 0, 'max_value': None},
            'collections_12_mths_ex_med': {'direction': 'decrease', 'max_change': 5, 'min_value': 0, 'max_value': None},

            # Financial capacity (improvable)
            'annual_inc': {'direction': 'increase', 'max_change': 50000, 'min_value': 0, 'max_value': 500000},
            'emp_length': {'direction': 'increase', 'max_change': 5, 'min_value': 0, 'max_value': 10},

            # Credit accounts (improvable over time)
            'open_acc': {'direction': 'both', 'max_change': 10, 'min_value': 0, 'max_value': 50},
            'total_acc': {'direction': 'increase', 'max_change': 10, 'min_value': 0, 'max_value': 100},

            # Loan characteristics (can be adjusted)
            'loan_amnt': {'direction': 'decrease', 'max_change': 10000, 'min_value': 1000, 'max_value': 35000},
            'term': {'direction': 'both', 'max_change': 1, 'min_value': 0, 'max_value': 1},  # Binary: 36 or 60 months
        }

        # Costs for each feature change (for prioritization)
        self.change_costs = {
            'fico_range_low': 10.0,       # High cost - takes time
            'fico_range_high': 10.0,
            'revol_util': 5.0,            # Medium cost - pay down debt
            'dti': 7.0,                   # Medium-high cost
            'delinq_2yrs': 15.0,          # Very high cost - time-based
            'inq_last_6mths': 8.0,        # High cost - wait 6 months
            'pub_rec': 20.0,              # Extremely high - hard to change
            'annual_inc': 8.0,            # High cost - get better job
            'loan_amnt': 2.0,             # Low cost - easy to adjust
            'emp_length': 12.0,           # Very high - time-based
            'open_acc': 6.0,              # Medium cost
            'total_acc': 6.0,
            'acc_now_delinq': 10.0,
            'collections_12_mths_ex_med': 15.0,
            'term': 1.0,                  # Very low cost
        }

    def is_mutable(self, feature: str) -> bool:
        """Check if a feature can be changed."""
        return feature not in self.immutable_features

    def get_constraints(self, feature: str) -> Optional[Dict]:
        """Get constraints for a specific feature."""
        return self.actionable_features.get(feature)

    def get_change_cost(self, feature: str, change_amount: float) -> float:
        """Calculate the cost of changing a feature."""
        if feature not in self.change_costs:
            return 0.0

        base_cost = self.change_costs[feature]
        # Cost scales with the magnitude of change
        normalized_change = abs(change_amount) / 100.0  # Normalize
        return base_cost * (1 + normalized_change)


class CounterfactualRecourseEngine:
    """Engine for generating counterfactual explanations."""

    def __init__(self, pd_threshold: float = 0.50):
        """
        Initialize the counterfactual recourse engine.

        Args:
            pd_threshold: PD threshold for loan approval (default: 50%)
        """
        self.pd_threshold = pd_threshold
        self.pd_models = {}
        self.pd_weights = None
        self.pd_model_names = None
        self.constraints = FeatureConstraints()
        self.feature_columns = None

    def load_champion_model(self):
        """Load the champion PD ensemble model."""
        print("\n" + "="*80)
        print("LOADING CHAMPION PD MODEL")
        print("="*80)

        # Model names and weights
        self.pd_model_names = [
            'logistic_regression',
            'random_forest',
            'xgboost',
            'neural_network'
        ]

        # Load weights
        weights_path = RESULTS_DIR / 'weighted_ensemble_metrics.pkl'
        if weights_path.exists():
            with open(weights_path, 'rb') as f:
                data = pickle.load(f)
                self.pd_weights = data.get('weights', np.array([0.25, 0.25, 0.25, 0.25]))
        else:
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

        # Adjust weights
        if len(models_loaded) < len(self.pd_model_names):
            loaded_indices = [i for i, name in enumerate(self.pd_model_names)
                            if name in models_loaded]
            self.pd_weights = self.pd_weights[loaded_indices]
            self.pd_weights = self.pd_weights / self.pd_weights.sum()
            self.pd_model_names = models_loaded

        print(f"\n[OK] Champion model loaded with {len(self.pd_models)} models")

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
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
                # Neural network needs scaling
                scaler_path = MODELS_DIR / 'neural_network_scaler.pkl'
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X_scaled = scaler.transform(X)
                    pred_proba = model.predict(X_scaled, verbose=0)
                    predictions.append(pred_proba.flatten())
                else:
                    pred_proba = model.predict(X.values, verbose=0)
                    predictions.append(pred_proba.flatten())
            else:
                pred_proba = model.predict_proba(X)[:, 1]
                predictions.append(pred_proba)

        # Weighted average
        predictions = np.array(predictions)
        pd_probs = np.average(predictions, axis=0, weights=self.pd_weights)

        return pd_probs

    def generate_counterfactual(
        self,
        original_instance: pd.Series,
        max_iterations: int = 1000,
        learning_rate: float = 0.1,
        diversity_weight: float = 0.5
    ) -> List[Dict]:
        """
        Generate counterfactual explanations for a rejected applicant.

        Args:
            original_instance: Original loan application (pd.Series)
            max_iterations: Maximum optimization iterations
            learning_rate: Step size for gradient-free optimization
            diversity_weight: Weight for diversity in counterfactuals

        Returns:
            List of counterfactual scenarios
        """
        print(f"\n  Generating counterfactual for applicant...")

        # Convert to DataFrame for prediction
        original_df = pd.DataFrame([original_instance])
        original_pd = self.predict_pd(original_df)[0]

        print(f"    Original PD: {original_pd:.2%} (Rejected - above {self.pd_threshold:.0%} threshold)")

        # If already approved, no counterfactual needed
        if original_pd < self.pd_threshold:
            print(f"    Already approved! No counterfactual needed.")
            return []

        # Generate multiple diverse counterfactuals
        counterfactuals = []

        # Strategy 1: Improve FICO score
        cf1 = self._generate_fico_improvement(original_instance, original_pd)
        if cf1:
            counterfactuals.append(cf1)

        # Strategy 2: Reduce DTI
        cf2 = self._generate_dti_reduction(original_instance, original_pd)
        if cf2:
            counterfactuals.append(cf2)

        # Strategy 3: Reduce loan amount
        cf3 = self._generate_loan_reduction(original_instance, original_pd)
        if cf3:
            counterfactuals.append(cf3)

        # Strategy 4: Improve credit utilization
        cf4 = self._generate_utilization_improvement(original_instance, original_pd)
        if cf4:
            counterfactuals.append(cf4)

        # Strategy 5: Combined improvements
        cf5 = self._generate_combined_improvement(original_instance, original_pd)
        if cf5:
            counterfactuals.append(cf5)

        # Rank by cost
        counterfactuals = sorted(counterfactuals, key=lambda x: x['total_cost'])

        return counterfactuals

    def _generate_fico_improvement(self, original: pd.Series, original_pd: float) -> Optional[Dict]:
        """Generate counterfactual by improving FICO score."""
        if 'fico_range_low' not in original.index:
            return None

        counterfactual = original.copy()
        changes = {}
        total_cost = 0.0

        # Improve FICO in increments
        for fico_increase in range(10, 101, 10):
            counterfactual['fico_range_low'] = min(original['fico_range_low'] + fico_increase, 850)
            if 'fico_range_high' in counterfactual.index:
                counterfactual['fico_range_high'] = min(original['fico_range_high'] + fico_increase, 850)

            # Check if approved
            cf_df = pd.DataFrame([counterfactual])
            new_pd = self.predict_pd(cf_df)[0]

            if new_pd < self.pd_threshold:
                changes['fico_range_low'] = fico_increase
                if 'fico_range_high' in counterfactual.index:
                    changes['fico_range_high'] = fico_increase
                total_cost = self.constraints.get_change_cost('fico_range_low', fico_increase)

                return {
                    'strategy': 'Improve FICO Score',
                    'counterfactual': counterfactual,
                    'changes': changes,
                    'new_pd': new_pd,
                    'total_cost': total_cost,
                    'description': f"Increase FICO score by {fico_increase} points to {int(counterfactual['fico_range_low'])}"
                }

        return None

    def _generate_dti_reduction(self, original: pd.Series, original_pd: float) -> Optional[Dict]:
        """Generate counterfactual by reducing DTI."""
        if 'dti' not in original.index or original['dti'] <= 5:
            return None

        counterfactual = original.copy()
        changes = {}
        total_cost = 0.0

        # Reduce DTI in increments
        for dti_reduction in range(2, 21, 2):
            new_dti = max(original['dti'] - dti_reduction, 0)
            counterfactual['dti'] = new_dti

            # Check if approved
            cf_df = pd.DataFrame([counterfactual])
            new_pd = self.predict_pd(cf_df)[0]

            if new_pd < self.pd_threshold:
                changes['dti'] = -dti_reduction
                total_cost = self.constraints.get_change_cost('dti', dti_reduction)

                return {
                    'strategy': 'Reduce Debt-to-Income Ratio',
                    'counterfactual': counterfactual,
                    'changes': changes,
                    'new_pd': new_pd,
                    'total_cost': total_cost,
                    'description': f"Reduce DTI by {dti_reduction:.1f}% to {new_dti:.1f}% (pay down debt or increase income)"
                }

        return None

    def _generate_loan_reduction(self, original: pd.Series, original_pd: float) -> Optional[Dict]:
        """Generate counterfactual by reducing loan amount."""
        if 'loan_amnt' not in original.index or original['loan_amnt'] <= 1000:
            return None

        counterfactual = original.copy()
        changes = {}
        total_cost = 0.0

        # Reduce loan amount in increments
        reduction_percentages = [5, 10, 15, 20, 25, 30]
        for reduction_pct in reduction_percentages:
            reduction_amount = original['loan_amnt'] * (reduction_pct / 100)
            new_loan_amnt = max(original['loan_amnt'] - reduction_amount, 1000)
            counterfactual['loan_amnt'] = new_loan_amnt

            # Check if approved
            cf_df = pd.DataFrame([counterfactual])
            new_pd = self.predict_pd(cf_df)[0]

            if new_pd < self.pd_threshold:
                changes['loan_amnt'] = -reduction_amount
                total_cost = self.constraints.get_change_cost('loan_amnt', reduction_amount)

                return {
                    'strategy': 'Reduce Loan Amount',
                    'counterfactual': counterfactual,
                    'changes': changes,
                    'new_pd': new_pd,
                    'total_cost': total_cost,
                    'description': f"Reduce loan amount by {reduction_pct}% to £{new_loan_amnt:,.0f}"
                }

        return None

    def _generate_utilization_improvement(self, original: pd.Series, original_pd: float) -> Optional[Dict]:
        """Generate counterfactual by improving credit utilization."""
        if 'revol_util' not in original.index or original['revol_util'] <= 10:
            return None

        counterfactual = original.copy()
        changes = {}
        total_cost = 0.0

        # Reduce utilization in increments
        for util_reduction in range(5, 51, 5):
            new_util = max(original['revol_util'] - util_reduction, 0)
            counterfactual['revol_util'] = new_util

            # Check if approved
            cf_df = pd.DataFrame([counterfactual])
            new_pd = self.predict_pd(cf_df)[0]

            if new_pd < self.pd_threshold:
                changes['revol_util'] = -util_reduction
                total_cost = self.constraints.get_change_cost('revol_util', util_reduction)

                return {
                    'strategy': 'Reduce Credit Utilization',
                    'counterfactual': counterfactual,
                    'changes': changes,
                    'new_pd': new_pd,
                    'total_cost': total_cost,
                    'description': f"Reduce revolving utilization by {util_reduction}% to {new_util:.1f}% (pay down credit cards)"
                }

        return None

    def _generate_combined_improvement(self, original: pd.Series, original_pd: float) -> Optional[Dict]:
        """Generate counterfactual with combined small improvements."""
        counterfactual = original.copy()
        changes = {}
        total_cost = 0.0

        # Small improvements across multiple features
        improvements = [
            ('fico_range_low', 20, 'increase'),
            ('fico_range_high', 20, 'increase'),
            ('dti', 5, 'decrease'),
            ('revol_util', 10, 'decrease'),
        ]

        for feature, change, direction in improvements:
            if feature not in counterfactual.index:
                continue

            if direction == 'increase':
                counterfactual[feature] = original[feature] + change
            else:
                counterfactual[feature] = max(original[feature] - change, 0)

            changes[feature] = change if direction == 'increase' else -change
            total_cost += self.constraints.get_change_cost(feature, change)

        # Check if approved
        cf_df = pd.DataFrame([counterfactual])
        new_pd = self.predict_pd(cf_df)[0]

        if new_pd < self.pd_threshold:
            return {
                'strategy': 'Combined Small Improvements',
                'counterfactual': counterfactual,
                'changes': changes,
                'new_pd': new_pd,
                'total_cost': total_cost,
                'description': "Multiple small improvements: FICO +20, DTI -5%, Utilization -10%"
            }

        return None

    def explain_counterfactuals(self, counterfactuals: List[Dict]) -> pd.DataFrame:
        """
        Create a readable explanation of counterfactual scenarios.

        Args:
            counterfactuals: List of counterfactual dictionaries

        Returns:
            DataFrame with explanations
        """
        if not counterfactuals:
            return pd.DataFrame()

        explanations = []

        for i, cf in enumerate(counterfactuals, 1):
            explanation = {
                'Rank': i,
                'Strategy': cf['strategy'],
                'Description': cf['description'],
                'New_PD': f"{cf['new_pd']:.2%}",
                'Cost_Score': f"{cf['total_cost']:.1f}",
                'Changes': ', '.join([f"{k}: {v:+.1f}" for k, v in cf['changes'].items()])
            }
            explanations.append(explanation)

        return pd.DataFrame(explanations)


def main():
    """Main execution: Generate counterfactuals for rejected applicants."""
    print("\n" + "="*80)
    print("COUNTERFACTUAL RECOURSE ENGINE")
    print("="*80)
    print("\nPhase 6: Generating actionable 'what-if' scenarios for rejected applicants\n")

    # Initialize engine
    engine = CounterfactualRecourseEngine(pd_threshold=0.50)

    # Load champion model
    engine.load_champion_model()

    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)

    test_path = DATA_DIR / 'test.csv'
    test_data = pd.read_csv(test_path)

    if 'default' in test_data.columns:
        test_data = test_data.drop('default', axis=1)

    print(f"\nLoaded {len(test_data):,} applicants")

    # Get predictions for all applicants
    print("\n" + "="*80)
    print("IDENTIFYING REJECTED APPLICANTS")
    print("="*80)

    pd_predictions = engine.predict_pd(test_data)
    rejected_mask = pd_predictions >= engine.pd_threshold

    rejected_applicants = test_data[rejected_mask]
    print(f"\nRejected applicants: {len(rejected_applicants):,} ({rejected_mask.sum()/len(test_data):.1%})")
    print(f"Approved applicants: {(~rejected_mask).sum():,} ({(~rejected_mask).sum()/len(test_data):.1%})")

    # Generate counterfactuals for a sample of rejected applicants
    print("\n" + "="*80)
    print("GENERATING COUNTERFACTUAL EXPLANATIONS")
    print("="*80)

    sample_size = min(100, len(rejected_applicants))  # Sample 100 rejected applicants
    sample_rejected = rejected_applicants.sample(n=sample_size, random_state=42)

    print(f"\nGenerating counterfactuals for {sample_size} rejected applicants...")

    all_counterfactuals = []

    for idx, (_, applicant) in enumerate(sample_rejected.iterrows(), 1):
        if idx <= 5 or idx % 20 == 0:  # Print progress
            print(f"  Processing applicant {idx}/{sample_size}...")

        counterfactuals = engine.generate_counterfactual(applicant)

        if counterfactuals:
            # Get best (lowest cost) counterfactual
            best_cf = counterfactuals[0]

            all_counterfactuals.append({
                'Applicant_ID': applicant.get('id', idx),
                'Original_PD': engine.predict_pd(pd.DataFrame([applicant]))[0],
                'Strategy': best_cf['strategy'],
                'Description': best_cf['description'],
                'New_PD': best_cf['new_pd'],
                'Cost_Score': best_cf['total_cost'],
                'Num_Changes': len(best_cf['changes']),
                'Changes': str(best_cf['changes'])
            })

    # Create results DataFrame
    results_df = pd.DataFrame(all_counterfactuals)

    print(f"\n[OK] Generated counterfactuals for {len(results_df)} applicants")

    # Summary statistics
    print("\n" + "="*80)
    print("COUNTERFACTUAL SUMMARY")
    print("="*80)

    if len(results_df) > 0:
        print(f"\nStrategy Distribution:")
        print(results_df['Strategy'].value_counts())

        print(f"\nCost Statistics:")
        print(f"  Mean cost: {results_df['Cost_Score'].mean():.1f}")
        print(f"  Median cost: {results_df['Cost_Score'].median():.1f}")
        print(f"  Min cost: {results_df['Cost_Score'].min():.1f}")
        print(f"  Max cost: {results_df['Cost_Score'].max():.1f}")

        print(f"\nPD Improvement:")
        print(f"  Mean PD reduction: {(results_df['Original_PD'] - results_df['New_PD']).mean():.2%}")
        print(f"  Mean new PD: {results_df['New_PD'].mean():.2%}")

        # Save results
        output_path = RESULTS_DIR / 'counterfactual_recourse_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to: {output_path}")

    print("\n" + "="*80)
    print("[COMPLETE] COUNTERFACTUAL RECOURSE ENGINE COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
