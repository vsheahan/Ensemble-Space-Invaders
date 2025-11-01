"""
End-to-End Training Pipeline

Trains the complete ensemble detection system:
1. VAE encoder (optional - can use pretrained)
2. Supervised classifier on latent features
3. Ensemble fusion and calibration
4. Comprehensive evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import numpy as np
from typing import Optional

from data_utils import PromptDataset, create_dummy_dataset, load_from_file, combine_datasets
from vae_encoder import VAEEncoder
from auxiliary_features import AuxiliaryFeatureExtractor
from latent_classifier import LatentClassifier
from ensemble_detector import EnsembleDetector
from evaluate import EvaluationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ensemble Prompt Injection Detector")

    # Data
    parser.add_argument('--mode', type=str, default='dummy', choices=['dummy', 'files'],
                        help='Dataset mode: dummy or files')
    parser.add_argument('--safe-train', type=str, help='Path to safe training prompts')
    parser.add_argument('--safe-val', type=str, help='Path to safe validation prompts')
    parser.add_argument('--attack-train', type=str, help='Path to attack training prompts')
    parser.add_argument('--attack-val', type=str, help='Path to attack validation prompts')
    parser.add_argument('--attack-test', type=str, help='Path to attack test prompts')
    parser.add_argument('--attack-type', type=str, default='jailbreak', help='Attack type label')

    # Dummy data settings
    parser.add_argument('--num-safe-train', type=int, default=200, help='Number of safe training prompts (dummy mode)')
    parser.add_argument('--num-attack-train', type=int, default=100, help='Number of attack training prompts (dummy mode)')
    parser.add_argument('--num-safe-val', type=int, default=50, help='Number of safe validation prompts (dummy mode)')
    parser.add_argument('--num-attack-val', type=int, default=25, help='Number of attack validation prompts (dummy mode)')

    # Model settings
    parser.add_argument('--model-name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model for feature extraction')
    parser.add_argument('--vae-model-path', type=str, help='Path to pretrained VAE (optional)')
    parser.add_argument('--latent-dim', type=int, default=128, help='VAE latent dimension')

    # Classifier settings
    parser.add_argument('--classifier-type', type=str, default='xgboost',
                        choices=['logistic', 'mlp', 'xgboost'], help='Classifier type')

    # Ensemble settings
    parser.add_argument('--fusion-method', type=str, default='stacking',
                        choices=['weighted', 'stacking'], help='Ensemble fusion method')
    parser.add_argument('--vae-weight', type=float, default=0.3, help='VAE weight (weighted fusion)')
    parser.add_argument('--classifier-weight', type=float, default=0.5, help='Classifier weight (weighted fusion)')
    parser.add_argument('--auxiliary-weight', type=float, default=0.2, help='Auxiliary weight (weighted fusion)')

    # Training
    parser.add_argument('--balance-classes', action='store_true', help='Balance training classes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    # Output
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory')

    return parser.parse_args()


def load_data(args):
    """Load training and validation data."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    if args.mode == 'dummy':
        # Create dummy datasets
        print("Creating dummy datasets...")

        train_dataset = create_dummy_dataset(
            num_safe=args.num_safe_train,
            num_attack=args.num_attack_train,
            random_state=42
        )

        val_dataset = create_dummy_dataset(
            num_safe=args.num_safe_val,
            num_attack=args.num_attack_val,
            random_state=43
        )

        test_dataset = create_dummy_dataset(
            num_safe=args.num_safe_val,
            num_attack=args.num_attack_val,
            random_state=44
        )

    elif args.mode == 'files':
        # Load from files
        print("Loading from files...")

        # Training data
        safe_train = load_from_file(args.safe_train, label=0)
        attack_train = load_from_file(args.attack_train, label=1, attack_type=args.attack_type)
        train_dataset = combine_datasets([safe_train, attack_train])

        # Validation data
        safe_val = load_from_file(args.safe_val, label=0)
        attack_val = load_from_file(args.attack_val, label=1, attack_type=args.attack_type)
        val_dataset = combine_datasets([safe_val, attack_val])

        # Test data (use validation if not provided)
        if args.attack_test:
            attack_test = load_from_file(args.attack_test, label=1, attack_type=args.attack_type)
            # Use same safe validation data
            test_dataset = combine_datasets([safe_val, attack_test])
        else:
            test_dataset = val_dataset

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Balance classes if requested
    if args.balance_classes:
        print("\nBalancing training classes...")
        train_dataset = train_dataset.balance_classes(method='undersample')

    # Print statistics
    print(f"\nTraining set: {len(train_dataset)} prompts")
    print(f"  Distribution: {train_dataset.get_class_distribution()}")

    print(f"\nValidation set: {len(val_dataset)} prompts")
    print(f"  Distribution: {val_dataset.get_class_distribution()}")

    print(f"\nTest set: {len(test_dataset)} prompts")
    print(f"  Distribution: {test_dataset.get_class_distribution()}")

    return train_dataset, val_dataset, test_dataset


def train_classifier(args, vae_encoder, aux_extractor, train_dataset, val_dataset):
    """Train supervised classifier on latent features."""
    print("\n" + "="*70)
    print("TRAINING SUPERVISED CLASSIFIER")
    print("="*70)

    # Extract features from training set
    print("\nExtracting training features...")
    train_latent = vae_encoder.extract_latent_features(
        train_dataset.prompts,
        batch_size=args.batch_size,
        return_reconstruction_loss=True
    )
    train_aux = aux_extractor.extract_features(train_dataset.prompts)

    # Combine features
    X_train = np.hstack([
        train_latent['latent_vectors'],
        train_latent['reconstruction_loss'].reshape(-1, 1),
        train_aux
    ])
    y_train = train_dataset.labels

    print(f"Training feature shape: {X_train.shape}")

    # Extract validation features
    print("\nExtracting validation features...")
    val_latent = vae_encoder.extract_latent_features(
        val_dataset.prompts,
        batch_size=args.batch_size,
        return_reconstruction_loss=True
    )
    val_aux = aux_extractor.extract_features(val_dataset.prompts)

    X_val = np.hstack([
        val_latent['latent_vectors'],
        val_latent['reconstruction_loss'].reshape(-1, 1),
        val_aux
    ])
    y_val = val_dataset.labels

    # Train classifier
    classifier = LatentClassifier(
        model_type=args.classifier_type,
        calibrate=True,
        random_state=42
    )

    classifier.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("\nValidation set performance:")
    val_metrics = classifier.evaluate(X_val, y_val)
    for metric_name, value in val_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return classifier


def build_ensemble(args, vae_encoder, classifier, aux_extractor, val_dataset):
    """Build and calibrate ensemble detector."""
    print("\n" + "="*70)
    print("BUILDING ENSEMBLE DETECTOR")
    print("="*70)

    # Initialize ensemble
    weights = {
        'vae': args.vae_weight,
        'classifier': args.classifier_weight,
        'auxiliary': args.auxiliary_weight
    }

    ensemble = EnsembleDetector(
        vae_encoder=vae_encoder,
        latent_classifier=classifier,
        auxiliary_extractor=aux_extractor,
        fusion_method=args.fusion_method,
        weights=weights
    )

    # Get validation prompts by class
    safe_val_prompts = [p for p, l in zip(val_dataset.prompts, val_dataset.labels) if l == 0]
    attack_val_prompts = [p for p, l in zip(val_dataset.prompts, val_dataset.labels) if l == 1]

    # Calibrate scores
    ensemble.calibrate_vae_scores(safe_val_prompts, attack_val_prompts)
    ensemble.calibrate_auxiliary_scores(safe_val_prompts, attack_val_prompts)

    # Train meta-classifier if using stacking
    if args.fusion_method == 'stacking':
        ensemble.train_meta_classifier(val_dataset.prompts, val_dataset.labels)

    # Find operating points
    print("\nComputing operating points...")
    operating_points = ensemble.get_operating_points(
        val_dataset.prompts,
        val_dataset.labels,
        fpr_targets=[0.001, 0.01, 0.05]
    )

    print("\nOperating Points (FPR -> Threshold):")
    for fpr, threshold in operating_points.items():
        print(f"  {fpr*100:.2f}% FPR: threshold = {threshold:.4f}")

    return ensemble, operating_points


def evaluate_ensemble(ensemble, test_dataset, evaluator):
    """Comprehensive evaluation of ensemble."""
    print("\n" + "="*70)
    print("EVALUATING ENSEMBLE DETECTOR")
    print("="*70)

    # Get predictions
    print("\nGenerating predictions...")
    y_proba, breakdown = ensemble.predict_proba(test_dataset.prompts, return_breakdown=True)
    y_pred = ensemble.predict(test_dataset.prompts)

    # Extract latent vectors for visualization
    latent_features = ensemble.vae_encoder.extract_latent_features(test_dataset.prompts)

    # Generate comprehensive report
    metrics = evaluator.generate_report(
        y_true=test_dataset.labels,
        y_pred=y_pred,
        y_proba=y_proba,
        latent_vectors=latent_features['latent_vectors'],
        attack_types=test_dataset.attack_types,
        report_name="ensemble_evaluation"
    )

    return metrics


def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    train_dataset, val_dataset, test_dataset = load_data(args)

    # Initialize components
    print("\n" + "="*70)
    print("INITIALIZING COMPONENTS")
    print("="*70)

    # VAE Encoder
    print("\nInitializing VAE encoder...")
    vae_encoder = VAEEncoder(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        vae_model_path=args.vae_model_path
    )

    # Auxiliary feature extractor
    print("\nInitializing auxiliary feature extractor...")
    aux_extractor = AuxiliaryFeatureExtractor(
        tokenizer_name=args.model_name
    )

    # Train classifier
    classifier = train_classifier(args, vae_encoder, aux_extractor, train_dataset, val_dataset)

    # Build ensemble
    ensemble, operating_points = build_ensemble(args, vae_encoder, classifier, aux_extractor, val_dataset)

    # Evaluate
    evaluator = EvaluationMetrics(save_dir=args.results_dir)
    metrics = evaluate_ensemble(ensemble, test_dataset, evaluator)

    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    vae_path = os.path.join(args.output_dir, 'vae_encoder.pth')
    classifier_path = os.path.join(args.output_dir, 'classifier.pkl')
    ensemble_path = os.path.join(args.output_dir, 'ensemble.pkl')

    vae_encoder.save_vae(vae_path)
    classifier.save_model(classifier_path)
    ensemble.save_ensemble(ensemble_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {args.output_dir}")
    print(f"Results saved to: {args.results_dir}")

    print("\n🎯 Key Metrics:")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  Recall @ 1% FPR: {metrics.get('recall_at_fpr_0.01', 0):.4f}")


if __name__ == "__main__":
    main()
