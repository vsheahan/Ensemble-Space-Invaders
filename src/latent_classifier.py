"""
Latent Classifier

Supervised classifiers trained on VAE latent features + auxiliary features
for prompt injection detection.

Supports: Logistic Regression, MLP, XGBoost
"""

import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pickle


class LatentClassifier:
    """
    Supervised classifier for prompt injection detection.

    Trains on combined features:
    - Latent vectors from VAE encoder
    - Auxiliary heuristic features
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        calibrate: bool = True,
        random_state: int = 42,
        **model_kwargs
    ):
        """
        Initialize classifier.

        Args:
            model_type: 'logistic', 'mlp', or 'xgboost'
            calibrate: Whether to calibrate probabilities
            random_state: Random seed
            **model_kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        # Initialize scaler
        self.scaler = StandardScaler()

        # Initialize model
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Create the base classifier."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                **self.model_kwargs
            )

        elif self.model_type == 'mlp':
            # Default MLP architecture
            default_kwargs = {
                'hidden_layer_sizes': (256, 128),
                'activation': 'relu',
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'alpha': 0.001  # L2 regularization
            }
            default_kwargs.update(self.model_kwargs)

            return MLPClassifier(
                random_state=self.random_state,
                **default_kwargs
            )

        elif self.model_type == 'xgboost':
            # Default XGBoost parameters
            default_kwargs = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            default_kwargs.update(self.model_kwargs)

            return xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                **default_kwargs
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - binary 0/1
            X_val: Optional validation features for calibration
            y_val: Optional validation labels for calibration
        """
        print(f"\nTraining {self.model_type} classifier...")
        print(f"Training samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if self.model_type == 'xgboost' and X_val is not None:
            # Use early stopping with validation set
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]

            self.model.fit(
                X_scaled, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_scaled, y)

        # Cross-validation score on training set (before calibration)
        print("Computing cross-validation score...")
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=3, scoring='roc_auc'
        )
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Calibrate if requested
        if self.calibrate and X_val is not None and y_val is not None:
            print("Calibrating classifier...")
            X_val_scaled = self.scaler.transform(X_val)

            # Use Platt scaling (sigmoid calibration)
            calibrated_model = CalibratedClassifierCV(
                self.model,
                method='sigmoid',
                cv='prefit'
            )
            calibrated_model.fit(X_val_scaled, y_val)
            self.model = calibrated_model

        self.is_fitted = True
        print("Training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict attack probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Attack probabilities (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (attack)
        return probas

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold

        Returns:
            Binary predictions (n_samples,)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier on test set.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score
        )

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'pr_auc': average_precision_score(y, y_proba)
        }

        return metrics

    def save_model(self, path: str):
        """Save classifier to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier.")

        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'calibrate': self.calibrate,
            'random_state': self.random_state,
            'model_kwargs': self.model_kwargs
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Classifier saved to {path}")

    def load_model(self, path: str):
        """Load classifier from disk."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.model = checkpoint['model']
        self.scaler = checkpoint['scaler']
        self.model_type = checkpoint['model_type']
        self.calibrate = checkpoint['calibrate']
        self.random_state = checkpoint['random_state']
        self.model_kwargs = checkpoint['model_kwargs']
        self.is_fitted = True

        print(f"Classifier loaded from {path}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (if available).

        Returns:
            Feature importances or None
        """
        if not self.is_fitted:
            return None

        if self.model_type == 'xgboost':
            return self.model.feature_importances_

        elif self.model_type == 'logistic':
            # Use absolute coefficient values
            return np.abs(self.model.coef_[0])

        else:
            # MLP doesn't have easily interpretable feature importance
            return None


if __name__ == "__main__":
    # Test
    print("Testing Latent Classifier...")

    from sklearn.datasets import make_classification

    # Generate dummy data
    X_train, y_train = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        random_state=42
    )

    X_val, y_val = make_classification(
        n_samples=100,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        random_state=43
    )

    X_test, y_test = make_classification(
        n_samples=100,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        random_state=44
    )

    # Test each model type
    for model_type in ['logistic', 'mlp', 'xgboost']:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()}")
        print('='*60)

        classifier = LatentClassifier(model_type=model_type, calibrate=True)
        classifier.fit(X_train, y_train, X_val, y_val)

        metrics = classifier.evaluate(X_test, y_test)
        print(f"\nTest Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # Test prediction
        probas = classifier.predict_proba(X_test[:5])
        print(f"\nSample predictions (probabilities): {probas}")

        # Feature importance
        importance = classifier.get_feature_importance()
        if importance is not None:
            print(f"Feature importance shape: {importance.shape}")
            print(f"Top 5 features: {np.argsort(importance)[-5:][::-1]}")

    print("\nLatent Classifier test complete!")
