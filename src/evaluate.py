"""
Evaluation Utilities

Comprehensive evaluation metrics and visualizations for
prompt injection detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import pandas as pd


class EvaluationMetrics:
    """Compute and visualize detection metrics."""

    def __init__(self, save_dir: str = './results'):
        """
        Initialize evaluator.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        attack_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            attack_types: Attack type labels for per-type metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Standard metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)

        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Confusion matrix components
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn

        # Recall at specific FPR thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        for target_fpr in [0.001, 0.01, 0.05]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                metrics[f'recall_at_fpr_{target_fpr}'] = tpr[idx[-1]]
            else:
                metrics[f'recall_at_fpr_{target_fpr}'] = 0.0

        # Per-attack-type metrics
        if attack_types is not None:
            metrics['per_attack_type'] = self._compute_per_attack_metrics(
                y_true, y_pred, y_proba, attack_types
            )

        return metrics

    def _compute_per_attack_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        attack_types: List[str]
    ) -> Dict:
        """Compute metrics per attack type."""
        per_type_metrics = {}

        # Get unique attack types (excluding None for safe prompts)
        unique_types = set([t for t in attack_types if t is not None])

        for attack_type in unique_types:
            # Filter to this attack type vs safe
            mask = np.array([
                (t == attack_type and y_true[i] == 1) or y_true[i] == 0
                for i, t in enumerate(attack_types)
            ])

            if np.sum(mask) == 0:
                continue

            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]
            y_proba_subset = y_proba[mask]

            per_type_metrics[attack_type] = {
                'accuracy': accuracy_score(y_true_subset, y_pred_subset),
                'precision': precision_score(y_true_subset, y_pred_subset, zero_division=0),
                'recall': recall_score(y_true_subset, y_pred_subset, zero_division=0),
                'f1': f1_score(y_true_subset, y_pred_subset, zero_division=0),
                'samples': np.sum(y_true_subset == 1)
            }

        return per_type_metrics

    def print_metrics(self, metrics: Dict, title: str = "Evaluation Metrics"):
        """Print metrics in a readable format."""
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print('='*70)

        print("\nOverall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:    {metrics['pr_auc']:.4f}")

        print("\nFalse Rates:")
        print(f"  FPR: {metrics['fpr']:.4f} ({metrics['fpr']*100:.2f}%)")
        print(f"  FNR: {metrics['fnr']:.4f} ({metrics['fnr']*100:.2f}%)")

        print("\nConfusion Matrix:")
        print(f"  TP: {metrics['true_positives']:>5}  |  FP: {metrics['false_positives']:>5}")
        print(f"  FN: {metrics['false_negatives']:>5}  |  TN: {metrics['true_negatives']:>5}")

        print("\nRecall at Target FPR:")
        for fpr_threshold in [0.001, 0.01, 0.05]:
            recall_val = metrics.get(f'recall_at_fpr_{fpr_threshold}', 0.0)
            print(f"  Recall @ {fpr_threshold*100:.2f}% FPR: {recall_val:.4f}")

        # Per-attack-type metrics
        if 'per_attack_type' in metrics:
            print("\nPer-Attack-Type Metrics:")
            for attack_type, attack_metrics in metrics['per_attack_type'].items():
                print(f"\n  {attack_type} ({attack_metrics['samples']} samples):")
                print(f"    Precision: {attack_metrics['precision']:.4f}")
                print(f"    Recall:    {attack_metrics['recall']:.4f}")
                print(f"    F1:        {attack_metrics['f1']:.4f}")

        print('='*70 + '\n')

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        # Mark operating points
        for target_fpr in [0.001, 0.01, 0.05]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                plt.plot(fpr[idx[-1]], tpr[idx[-1]], 'ro', markersize=8)
                plt.annotate(
                    f'FPR={target_fpr:.1%}\nTPR={tpr[idx[-1]]:.1%}',
                    xy=(fpr[idx[-1]], tpr[idx[-1]]),
                    xytext=(10, -10),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
                )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = f"{self.save_dir}/roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curve saved to {save_path}")

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')

        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Random (={baseline:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if save_path is None:
            save_path = f"{self.save_dir}/precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Precision-Recall curve saved to {save_path}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Attack'],
            yticklabels=['Safe', 'Attack']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)

        if save_path is None:
            save_path = f"{self.save_dir}/confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {save_path}")

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        save_path: Optional[str] = None
    ):
        """Plot calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')

        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = f"{self.save_dir}/calibration_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Calibration curve saved to {save_path}")

    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Score Distribution",
        save_path: Optional[str] = None
    ):
        """Plot distribution of scores by class."""
        plt.figure(figsize=(10, 5))

        # Safe prompts
        safe_scores = y_proba[y_true == 0]
        plt.hist(safe_scores, bins=50, alpha=0.6, label='Safe', color='blue')

        # Attack prompts
        attack_scores = y_proba[y_true == 1]
        plt.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red')

        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = f"{self.save_dir}/score_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Score distribution saved to {save_path}")

    def plot_latent_space_tsne(
        self,
        latent_vectors: np.ndarray,
        labels: np.ndarray,
        title: str = "Latent Space (t-SNE)",
        save_path: Optional[str] = None
    ):
        """Visualize latent space using t-SNE."""
        print("Computing t-SNE embedding...")

        # Subsample if too large
        if len(latent_vectors) > 2000:
            indices = np.random.choice(len(latent_vectors), 2000, replace=False)
            latent_vectors = latent_vectors[indices]
            labels = labels[indices]

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(latent_vectors)

        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        for label, color, name in zip([0, 1], colors, ['Safe', 'Attack']):
            mask = labels == label
            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=color,
                alpha=0.6,
                s=20,
                label=name
            )

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = f"{self.save_dir}/latent_space_tsne.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"t-SNE plot saved to {save_path}")

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        latent_vectors: Optional[np.ndarray] = None,
        attack_types: Optional[List[str]] = None,
        report_name: str = "evaluation_report"
    ):
        """Generate comprehensive evaluation report."""
        print(f"\nGenerating evaluation report: {report_name}")

        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_proba, attack_types)

        # Print metrics
        self.print_metrics(metrics, title=f"{report_name} - Metrics")

        # Generate plots
        self.plot_roc_curve(y_true, y_proba, save_path=f"{self.save_dir}/{report_name}_roc.png")
        self.plot_precision_recall_curve(y_true, y_proba, save_path=f"{self.save_dir}/{report_name}_pr.png")
        self.plot_confusion_matrix(y_true, y_pred, save_path=f"{self.save_dir}/{report_name}_cm.png")
        self.plot_calibration_curve(y_true, y_proba, save_path=f"{self.save_dir}/{report_name}_calibration.png")
        self.plot_score_distribution(y_true, y_proba, save_path=f"{self.save_dir}/{report_name}_scores.png")

        if latent_vectors is not None:
            self.plot_latent_space_tsne(
                latent_vectors, y_true,
                save_path=f"{self.save_dir}/{report_name}_tsne.png"
            )

        # Save metrics to JSON
        import json

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        metrics_serializable = convert_to_serializable(metrics)

        with open(f"{self.save_dir}/{report_name}_metrics.json", 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"\nReport generation complete! Files saved to {self.save_dir}/")

        return metrics


if __name__ == "__main__":
    print("Evaluation utilities module defined.")
    print("Use with trained models to generate reports.")
