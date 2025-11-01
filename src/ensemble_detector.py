"""
Ensemble Detector

Fuses scores from multiple detectors:
1. VAE reconstruction error (unsupervised)
2. Supervised classifier probability
3. Auxiliary heuristic features

Supports weighted averaging and logistic stacking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pickle

from vae_encoder import VAEEncoder
from latent_classifier import LatentClassifier
from auxiliary_features import AuxiliaryFeatureExtractor


class EnsembleDetector:
    """
    Ensemble prompt injection detector.

    Combines multiple detection signals for robust performance.
    """

    def __init__(
        self,
        vae_encoder: VAEEncoder,
        latent_classifier: LatentClassifier,
        auxiliary_extractor: AuxiliaryFeatureExtractor,
        fusion_method: str = 'weighted',
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize ensemble detector.

        Args:
            vae_encoder: VAE encoder for latent features and reconstruction loss
            latent_classifier: Supervised classifier on latent+auxiliary features
            auxiliary_extractor: Auxiliary feature extractor
            fusion_method: 'weighted' or 'stacking'
            weights: Fusion weights {'vae': w1, 'classifier': w2, 'auxiliary': w3}
            threshold: Detection threshold
        """
        self.vae_encoder = vae_encoder
        self.latent_classifier = latent_classifier
        self.auxiliary_extractor = auxiliary_extractor

        self.fusion_method = fusion_method
        self.threshold = threshold

        # Default weights
        if weights is None:
            self.weights = {
                'vae': 0.3,
                'classifier': 0.5,
                'auxiliary': 0.2
            }
        else:
            self.weights = weights

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        # Meta-classifier for stacking (trained later)
        self.meta_classifier = None
        self.meta_scaler = StandardScaler()

        # Statistics for VAE score normalization
        self.vae_score_mean = 0.0
        self.vae_score_std = 1.0

        # Statistics for auxiliary score normalization
        self.aux_score_mean = 0.0
        self.aux_score_std = 1.0

    def compute_vae_score(self, prompts: List[str]) -> np.ndarray:
        """
        Compute VAE reconstruction error score.

        Args:
            prompts: List of text prompts

        Returns:
            Normalized reconstruction scores (n_prompts,)
        """
        # Compute reconstruction loss
        recon_losses = self.vae_encoder.compute_reconstruction_loss(prompts)

        # Normalize using statistics
        scores = (recon_losses - self.vae_score_mean) / (self.vae_score_std + 1e-8)

        # Apply sigmoid to get [0, 1] range
        scores = 1.0 / (1.0 + np.exp(-scores))

        return scores

    def compute_auxiliary_score(self, prompts: List[str]) -> np.ndarray:
        """
        Compute auxiliary heuristic score.

        Args:
            prompts: List of text prompts

        Returns:
            Auxiliary scores (n_prompts,)
        """
        # Extract features
        features = self.auxiliary_extractor.extract_features(prompts)

        # Simple scoring: normalize and average
        # Features that tend to be higher in attacks: length, caps ratio, special tokens
        # We'll use a simple weighted sum of normalized features

        # Normalize features
        features_normalized = (features - self.aux_score_mean) / (self.aux_score_std + 1e-8)

        # Average across features
        scores = np.mean(features_normalized, axis=1)

        # Sigmoid to [0, 1]
        scores = 1.0 / (1.0 + np.exp(-scores))

        return scores

    def compute_classifier_score(
        self,
        prompts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute supervised classifier score.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for feature extraction

        Returns:
            Attack probabilities (n_prompts,)
        """
        # Extract latent + auxiliary features
        latent_features = self.vae_encoder.extract_latent_features(
            prompts, batch_size=batch_size, return_reconstruction_loss=True
        )
        auxiliary_features = self.auxiliary_extractor.extract_features(prompts)

        # Combine features
        combined_features = np.hstack([
            latent_features['latent_vectors'],
            latent_features['reconstruction_loss'].reshape(-1, 1),
            auxiliary_features
        ])

        # Predict with classifier
        probas = self.latent_classifier.predict_proba(combined_features)

        return probas

    def predict_proba(
        self,
        prompts: List[str],
        return_breakdown: bool = False
    ) -> np.ndarray:
        """
        Predict attack probabilities using ensemble.

        Args:
            prompts: List of text prompts
            return_breakdown: If True, return individual scores

        Returns:
            If return_breakdown=False: Final probabilities (n_prompts,)
            If return_breakdown=True: (probabilities, breakdown_dict)
        """
        # Compute individual scores
        vae_scores = self.compute_vae_score(prompts)
        classifier_scores = self.compute_classifier_score(prompts)
        auxiliary_scores = self.compute_auxiliary_score(prompts)

        # Fuse scores
        if self.fusion_method == 'weighted':
            # Weighted average
            final_scores = (
                self.weights['vae'] * vae_scores +
                self.weights['classifier'] * classifier_scores +
                self.weights['auxiliary'] * auxiliary_scores
            )

        elif self.fusion_method == 'stacking':
            # Logistic stacking
            if self.meta_classifier is None:
                raise RuntimeError(
                    "Meta-classifier not trained. Call train_meta_classifier() first "
                    "or use fusion_method='weighted'."
                )

            # Stack scores
            stacked_features = np.column_stack([vae_scores, classifier_scores, auxiliary_scores])
            stacked_features_scaled = self.meta_scaler.transform(stacked_features)

            # Predict with meta-classifier
            final_scores = self.meta_classifier.predict_proba(stacked_features_scaled)[:, 1]

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        if return_breakdown:
            breakdown = {
                'vae_score': vae_scores,
                'classifier_score': classifier_scores,
                'auxiliary_score': auxiliary_scores,
                'final_score': final_scores
            }
            return final_scores, breakdown
        else:
            return final_scores

    def predict(
        self,
        prompts: List[str],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            prompts: List of text prompts
            threshold: Detection threshold (uses self.threshold if None)

        Returns:
            Binary predictions (n_prompts,)
        """
        if threshold is None:
            threshold = self.threshold

        probas = self.predict_proba(prompts)
        return (probas >= threshold).astype(int)

    def predict_single(
        self,
        prompt: str,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Predict for a single prompt with detailed breakdown.

        Args:
            prompt: Text prompt
            threshold: Detection threshold

        Returns:
            Dictionary with prediction details
        """
        if threshold is None:
            threshold = self.threshold

        probas, breakdown = self.predict_proba([prompt], return_breakdown=True)

        result = {
            'is_attack': bool(probas[0] >= threshold),
            'probability': float(probas[0]),
            'threshold': threshold,
            'vae_score': float(breakdown['vae_score'][0]),
            'classifier_score': float(breakdown['classifier_score'][0]),
            'auxiliary_score': float(breakdown['auxiliary_score'][0]),
            'weights': self.weights,
            'fusion_method': self.fusion_method
        }

        return result

    def calibrate_vae_scores(
        self,
        safe_prompts: List[str],
        attack_prompts: List[str]
    ):
        """
        Calibrate VAE score normalization using validation data.

        Args:
            safe_prompts: Safe validation prompts
            attack_prompts: Attack validation prompts
        """
        print("Calibrating VAE scores...")

        all_prompts = safe_prompts + attack_prompts
        recon_losses = self.vae_encoder.compute_reconstruction_loss(all_prompts)

        self.vae_score_mean = np.mean(recon_losses)
        self.vae_score_std = np.std(recon_losses)

        print(f"VAE score mean: {self.vae_score_mean:.6f}, std: {self.vae_score_std:.6f}")

    def calibrate_auxiliary_scores(
        self,
        safe_prompts: List[str],
        attack_prompts: List[str]
    ):
        """
        Calibrate auxiliary score normalization.

        Args:
            safe_prompts: Safe validation prompts
            attack_prompts: Attack validation prompts
        """
        print("Calibrating auxiliary scores...")

        all_prompts = safe_prompts + attack_prompts
        features = self.auxiliary_extractor.extract_features(all_prompts)

        self.aux_score_mean = np.mean(features, axis=0)
        self.aux_score_std = np.std(features, axis=0)

        print(f"Auxiliary score calibrated using {len(all_prompts)} prompts.")

    def train_meta_classifier(
        self,
        val_prompts: List[str],
        val_labels: np.ndarray
    ):
        """
        Train meta-classifier for stacking fusion.

        Args:
            val_prompts: Validation prompts
            val_labels: Validation labels
        """
        print("Training meta-classifier for stacking...")

        # Compute individual scores
        vae_scores = self.compute_vae_score(val_prompts)
        classifier_scores = self.compute_classifier_score(val_prompts)
        auxiliary_scores = self.compute_auxiliary_score(val_prompts)

        # Stack features
        stacked_features = np.column_stack([vae_scores, classifier_scores, auxiliary_scores])

        # Scale
        stacked_features_scaled = self.meta_scaler.fit_transform(stacked_features)

        # Train logistic regression
        self.meta_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_classifier.fit(stacked_features_scaled, val_labels)

        # Calibrate
        self.meta_classifier = CalibratedClassifierCV(
            self.meta_classifier,
            method='sigmoid',
            cv='prefit'
        )
        self.meta_classifier.fit(stacked_features_scaled, val_labels)

        print("Meta-classifier trained!")

    def set_threshold(self, threshold: float):
        """Set detection threshold."""
        self.threshold = threshold

    def get_operating_points(
        self,
        val_prompts: List[str],
        val_labels: np.ndarray,
        fpr_targets: List[float] = [0.001, 0.01, 0.05]
    ) -> Dict[float, float]:
        """
        Get operating point thresholds for target FPRs.

        Args:
            val_prompts: Validation prompts
            val_labels: Validation labels
            fpr_targets: Target false positive rates

        Returns:
            Dictionary mapping FPR -> threshold
        """
        from sklearn.metrics import roc_curve

        probas = self.predict_proba(val_prompts)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(val_labels, probas)

        # Find thresholds for target FPRs
        operating_points = {}
        for target_fpr in fpr_targets:
            # Find largest threshold with FPR <= target
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                operating_points[target_fpr] = thresholds[idx[-1]]
            else:
                operating_points[target_fpr] = 1.0  # No threshold achieves this FPR

        return operating_points

    def save_ensemble(self, path: str):
        """Save ensemble configuration."""
        checkpoint = {
            'fusion_method': self.fusion_method,
            'weights': self.weights,
            'threshold': self.threshold,
            'vae_score_mean': self.vae_score_mean,
            'vae_score_std': self.vae_score_std,
            'aux_score_mean': self.aux_score_mean,
            'aux_score_std': self.aux_score_std,
            'meta_classifier': self.meta_classifier,
            'meta_scaler': self.meta_scaler
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Ensemble saved to {path}")

    def load_ensemble(self, path: str):
        """Load ensemble configuration."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.fusion_method = checkpoint['fusion_method']
        self.weights = checkpoint['weights']
        self.threshold = checkpoint['threshold']
        self.vae_score_mean = checkpoint['vae_score_mean']
        self.vae_score_std = checkpoint['vae_score_std']
        self.aux_score_mean = checkpoint['aux_score_mean']
        self.aux_score_std = checkpoint['aux_score_std']
        self.meta_classifier = checkpoint['meta_classifier']
        self.meta_scaler = checkpoint['meta_scaler']

        print(f"Ensemble loaded from {path}")


if __name__ == "__main__":
    print("Ensemble Detector module defined.")
    print("Use training scripts to create and test ensemble detector.")
