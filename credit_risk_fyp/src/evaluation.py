"""
Model Evaluation Module for Credit Risk Assessment FYP
Comprehensive evaluation metrics and visualizations for all models
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .config import FIGURES_DIR, REPORTS_DIR, VISUALIZATION_CONFIG

# Setup logger
logger = logging.getLogger('credit_risk_fyp.evaluation')


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.

    Features:
    - Multiple classification metrics
    - ROC and PR curves
    - Confusion matrices
    - Threshold optimization
    - Model comparison
    - Report generation
    """

    def __init__(
        self,
        figures_dir: Union[str, Path] = None,
        reports_dir: Union[str, Path] = None,
        figsize: Tuple[int, int] = None,
        dpi: int = None
    ):
        """
        Initialize ModelEvaluator.

        Args:
            figures_dir: Directory to save figures
            reports_dir: Directory to save reports
            figsize: Figure size for plots
            dpi: DPI for saved figures
        """
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.figsize = figsize or VISUALIZATION_CONFIG['figure_size']
        self.dpi = dpi or VISUALIZATION_CONFIG['dpi']

        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        try:
            plt.style.use(VISUALIZATION_CONFIG['style'])
        except:
            plt.style.use('seaborn-v0_8-darkgrid')

        sns.set_palette(VISUALIZATION_CONFIG['color_palette'])

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Advanced metrics
        metrics['matthews_corr_coef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Probability-based metrics (if probabilities provided)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)

            # PR AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)

        return metrics

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Complete evaluation of a model.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            model_name: Name of the model

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics['threshold'] = threshold
        metrics['model_name'] = model_name

        # Add sample count information
        metrics['n_samples'] = len(y_true)
        metrics['n_positive'] = int(np.sum(y_true))
        metrics['n_negative'] = int(len(y_true) - np.sum(y_true))

        # Log key metrics
        logger.info(f"{model_name} Results:")
        logger.info(f"  AUC-ROC: {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

        return metrics

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            save_path = self.figures_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot PR curve
        ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AUC = {pr_auc:.4f})')

        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline ({baseline:.2f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
            save_path = self.figures_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        normalize: bool = False,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            normalize: Whether to normalize values
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = '(Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name} {title_suffix}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            norm_str = '_normalized' if normalize else ''
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}{norm_str}.png'
            save_path = self.figures_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot how metrics change with threshold.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        thresholds = np.linspace(0, 1, 100)
        metrics_over_threshold = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics_over_threshold['precision'].append(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics_over_threshold['recall'].append(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics_over_threshold['f1_score'].append(
                f1_score(y_true, y_pred, zero_division=0)
            )
            metrics_over_threshold['accuracy'].append(
                accuracy_score(y_true, y_pred)
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(thresholds, metrics_over_threshold['precision'],
                label='Precision', linewidth=2)
        ax.plot(thresholds, metrics_over_threshold['recall'],
                label='Recall', linewidth=2)
        ax.plot(thresholds, metrics_over_threshold['f1_score'],
                label='F1-Score', linewidth=2)
        ax.plot(thresholds, metrics_over_threshold['accuracy'],
                label='Accuracy', linewidth=2)

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Metrics vs Threshold - {model_name}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            filename = f'threshold_analysis_{model_name.lower().replace(" ", "_")}.png'
            save_path = self.figures_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")

        return fig

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a given metric.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')

        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        thresholds = np.linspace(0, 1, 100)
        best_score = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f"Optimal threshold for {metric}: {best_threshold:.4f} "
                   f"(score: {best_score:.4f})")

        return best_threshold, best_score

    def calibrate_probabilities(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = 'isotonic',
        cv: int = 5
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calibrate predicted probabilities for better reliability.

        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated predicted probabilities
            method: Calibration method ('sigmoid' for Platt scaling, 'isotonic' for isotonic regression)
            cv: Number of cross-validation folds (not used for direct calibration)

        Returns:
            Tuple of (calibrated_probabilities, calibration_metrics)
        """
        from sklearn.calibration import calibration_curve
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        logger.info(f"Calibrating probabilities using {method} method...")

        # Store original metrics
        original_brier = brier_score_loss(y_true, y_pred_proba)
        original_log_loss = log_loss(y_true, y_pred_proba)

        # Apply calibration
        if method == 'sigmoid':
            # Platt scaling (logistic regression)
            calibrator = LogisticRegression()
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
            y_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        elif method == 'isotonic':
            # Isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            y_calibrated = calibrator.fit_transform(y_pred_proba, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {method}. Use 'sigmoid' or 'isotonic'")

        # Calculate calibrated metrics
        calibrated_brier = brier_score_loss(y_true, y_calibrated)
        calibrated_log_loss = log_loss(y_true, y_calibrated)

        # Calculate calibration curve metrics
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_calibrated, n_bins=10)

        # Expected Calibration Error (ECE)
        ece_before = np.mean(np.abs(prob_true - prob_pred))
        ece_after = np.mean(np.abs(prob_true_cal - prob_pred_cal))

        calibration_metrics = {
            'method': method,
            'brier_score_before': original_brier,
            'brier_score_after': calibrated_brier,
            'brier_improvement': original_brier - calibrated_brier,
            'log_loss_before': original_log_loss,
            'log_loss_after': calibrated_log_loss,
            'log_loss_improvement': original_log_loss - calibrated_log_loss,
            'ece_before': ece_before,
            'ece_after': ece_after,
            'ece_improvement': ece_before - ece_after
        }

        logger.info(f"Calibration complete:")
        logger.info(f"  Brier score: {original_brier:.4f} → {calibrated_brier:.4f} "
                   f"(improvement: {calibration_metrics['brier_improvement']:.4f})")
        logger.info(f"  Log loss: {original_log_loss:.4f} → {calibrated_log_loss:.4f} "
                   f"(improvement: {calibration_metrics['log_loss_improvement']:.4f})")
        logger.info(f"  ECE: {ece_before:.4f} → {ece_after:.4f} "
                   f"(improvement: {calibration_metrics['ece_improvement']:.4f})")

        return y_calibrated, calibration_metrics

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_calibrated_proba: np.ndarray = None,
        model_name: str = "Model",
        n_bins: int = 10,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot calibration curve showing probability calibration.

        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated predicted probabilities
            y_calibrated_proba: Calibrated probabilities (optional)
            model_name: Name of the model
            n_bins: Number of bins for calibration curve
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

        # Plot uncalibrated curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
                label=f'Uncalibrated (Brier: {brier_score_loss(y_true, y_pred_proba):.4f})')

        # Plot calibrated curve if provided
        if y_calibrated_proba is not None:
            prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_calibrated_proba, n_bins=n_bins)
            ax.plot(prob_pred_cal, prob_true_cal, marker='s', linewidth=2,
                   label=f'Calibrated (Brier: {brier_score_loss(y_true, y_calibrated_proba):.4f})')

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            filename = f'calibration_curve_{model_name.lower().replace(" ", "_")}.png'
            save_path = self.figures_dir / filename
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")

        return fig

    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Create comparison plot for multiple models.

        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            metrics: List of metrics to compare
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']

        # Prepare data for plotting
        model_names = list(results_dict.keys())

        # Create DataFrame for easier plotting
        data = []
        for model_name, model_metrics in results_dict.items():
            for metric in metrics:
                if metric in model_metrics:
                    data.append({
                        'Model': model_name,
                        'Metric': metric.upper().replace('_', ' '),
                        'Score': model_metrics[metric]
                    })

        df = pd.DataFrame(data)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Group by metric
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            model_data = df[df['Model'] == model_name]
            scores = [model_data[model_data['Metric'] == m.upper().replace('_', ' ')]['Score'].values[0]
                     if len(model_data[model_data['Metric'] == m.upper().replace('_', ' ')]) > 0 else 0
                     for m in metrics]

            ax.bar(x + i * width, scores, width, label=model_name)

        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if save:
            save_path = self.figures_dir / 'model_comparison.png'
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")

        return fig

    def generate_report(
        self,
        results_dict: Dict[str, Dict[str, float]],
        output_path: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.

        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            output_path: Path to save report (CSV)

        Returns:
            DataFrame with all results
        """
        df = pd.DataFrame(results_dict).T

        # Sort by AUC-ROC if available
        if 'roc_auc' in df.columns:
            df = df.sort_values('roc_auc', ascending=False)

        # Format columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['true_positives', 'true_negatives',
                          'false_positives', 'false_negatives']:
                df[col] = df[col].round(4)

        # Save to CSV if path provided
        if output_path:
            output_path = Path(output_path)
            df.to_csv(output_path)
            logger.info(f"Report saved to {output_path}")

        return df

    def evaluate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        threshold: float = 0.5,
        save: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate all evaluation plots for a model.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            threshold: Classification threshold
            save: Whether to save figures

        Returns:
            Dictionary of figure objects
        """
        logger.info(f"Generating all evaluation plots for {model_name}...")

        y_pred = (y_pred_proba >= threshold).astype(int)

        figures = {}

        # ROC Curve
        figures['roc'] = self.plot_roc_curve(y_true, y_pred_proba, model_name, save)

        # PR Curve
        figures['pr'] = self.plot_pr_curve(y_true, y_pred_proba, model_name, save)

        # Confusion Matrix
        figures['cm'] = self.plot_confusion_matrix(y_true, y_pred, model_name, False, save)
        figures['cm_normalized'] = self.plot_confusion_matrix(y_true, y_pred, model_name, True, save)

        # Threshold Analysis
        figures['threshold'] = self.plot_threshold_analysis(y_true, y_pred_proba, model_name, save)

        logger.info(f"All plots generated for {model_name}")

        return figures


def quick_evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
    show_plots: bool = True
) -> Dict[str, float]:
    """
    Quick evaluation function for convenience.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        threshold: Classification threshold
        show_plots: Whether to display plots

    Returns:
        Dictionary of metrics
    """
    evaluator = ModelEvaluator()

    # Calculate metrics
    metrics = evaluator.evaluate(y_true, y_pred_proba, threshold, model_name)

    # Generate plots
    if show_plots:
        evaluator.evaluate_all_plots(y_true, y_pred_proba, model_name, threshold)
        plt.show()

    return metrics
if __name__ == "__main__":
    """
    Test evaluation module with REAL credit risk data
    """
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    from src.config import RAW_DATA_DIR, DATASET_CONFIG
    from src.data_loader import load_data
    from src.preprocessor import DataPreprocessor
    from src.models.xgboost_model import XGBoostModel

    print("=" * 80)
    print("EVALUATION MODULE TEST - USING REAL DATA")
    print("=" * 80)

    # Load REAL training data
    print("\n[1/4] Loading REAL training data...")
    train_path = RAW_DATA_DIR / DATASET_CONFIG['train_dataset']
    df = load_data(train_path, optimize=True, nrows=15000)  # Use 15k for test
    print(f"Loaded {len(df):,} rows from: {DATASET_CONFIG['train_dataset']}")

    # Prepare data
    target_col = DATASET_CONFIG['target_column']
    id_col = DATASET_CONFIG['id_column']
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    # Preprocess
    print("\n[2/4] Preprocessing data...")
    preprocessor = DataPreprocessor()

    # Combine X_train and y_train for preprocessing
    train_df_combined = X_train.copy()
    train_df_combined[target_col] = y_train

    X_train_processed, _ = preprocessor.fit_transform(train_df_combined)
    X_test_processed = preprocessor.transform(X_test)

    # Train quick model
    print("\n[3/4] Training quick XGBoost model...")
    model = XGBoostModel(params={
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 30,
        'tree_method': 'hist',
        'random_state': 42
    })

    # Use 80/20 split for train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    model.train(X_tr, y_tr, X_val, y_val)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test_processed)

    # Evaluate
    print("\n[4/4] Evaluating model on test set...")
    evaluator = ModelEvaluator()

    metrics = evaluator.evaluate(y_test, y_pred_proba, model_name="Test_XGBoost")
    evaluator.evaluate_all_plots(y_test, y_pred_proba, model_name="Test_XGBoost")

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS (TEST SET - REAL DATA)")
    print("=" * 80)
    print(f"\nDataset: {DATASET_CONFIG['train_dataset']}")
    print(f"Total samples evaluated: {metrics['n_samples']:,}")
    print(f"Positive class (defaults): {metrics['n_positive']:,}")
    print(f"Negative class (no defaults): {metrics['n_negative']:,}")

    print(f"\nMetrics:")
    print(f"AUC-ROC:     {metrics['roc_auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")

    print(f"\nPlots saved to: {evaluator.figures_dir}")
    print("=" * 80)
    print("SUCCESS! Evaluation module working with REAL credit risk data.")
    print("=" * 80)