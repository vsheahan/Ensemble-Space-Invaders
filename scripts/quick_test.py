"""
Quick Test Script

Tests individual components to verify the installation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

print("="*70)
print("ENSEMBLE SPACE INVADERS - COMPONENT TESTS")
print("="*70)

# Test 1: Data utilities
print("\n[1/5] Testing Data Utilities...")
try:
    from data_utils import create_dummy_dataset
    dataset = create_dummy_dataset(num_safe=10, num_attack=5)
    print(f"✅ Data utilities working. Created {len(dataset)} prompts.")
except Exception as e:
    print(f"❌ Data utilities failed: {e}")
    sys.exit(1)

# Test 2: Auxiliary features
print("\n[2/5] Testing Auxiliary Feature Extractor...")
try:
    from auxiliary_features import AuxiliaryFeatureExtractor
    extractor = AuxiliaryFeatureExtractor()
    features = extractor.extract_features(dataset.prompts[:3])
    print(f"✅ Auxiliary features working. Shape: {features.shape}")
except Exception as e:
    print(f"❌ Auxiliary features failed: {e}")
    sys.exit(1)

# Test 3: VAE encoder (this may take a moment to download TinyLlama)
print("\n[3/5] Testing VAE Encoder (may download model on first run)...")
try:
    from vae_encoder import VAEEncoder
    vae = VAEEncoder(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", latent_dim=128)
    print(f"✅ VAE encoder initialized. Latent dim: {vae.get_feature_dim()}")

    # Quick feature extraction test
    print("   Testing feature extraction on 2 prompts...")
    test_prompts = dataset.prompts[:2]
    features = vae.extract_latent_features(test_prompts)
    print(f"✅ Feature extraction working. Latent vectors shape: {features['latent_vectors'].shape}")
except Exception as e:
    print(f"❌ VAE encoder failed: {e}")
    print("   Note: This requires latent-space-invaders to be cloned alongside this project.")
    sys.exit(1)

# Test 4: Latent classifier
print("\n[4/5] Testing Latent Classifier...")
try:
    from latent_classifier import LatentClassifier
    import numpy as np

    # Create dummy training data
    X_train = np.random.randn(50, 100)
    y_train = np.random.randint(0, 2, 50)

    classifier = LatentClassifier(model_type='logistic')
    classifier.fit(X_train, y_train)

    # Test prediction
    X_test = np.random.randn(10, 100)
    probas = classifier.predict_proba(X_test)

    print(f"✅ Latent classifier working. Predictions shape: {probas.shape}")
except Exception as e:
    print(f"❌ Latent classifier failed: {e}")
    sys.exit(1)

# Test 5: Evaluation utilities
print("\n[5/5] Testing Evaluation Utilities...")
try:
    from evaluate import EvaluationMetrics
    import numpy as np

    evaluator = EvaluationMetrics(save_dir='./test_results')

    # Create dummy evaluation data
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])

    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)

    print(f"✅ Evaluation utilities working.")
    print(f"   Sample metrics: Accuracy={metrics['accuracy']:.3f}, "
          f"ROC AUC={metrics['roc_auc']:.3f}")

    # Clean up test directory
    import shutil
    if os.path.exists('./test_results'):
        shutil.rmtree('./test_results')

except Exception as e:
    print(f"❌ Evaluation utilities failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL COMPONENT TESTS PASSED!")
print("="*70)
print("\nYou're ready to train the ensemble detector!")
print("\nNext steps:")
print("  1. Run quick training on dummy data:")
print("     python scripts/train_pipeline.py --mode dummy")
print("\n  2. Or train on your own datasets:")
print("     python scripts/train_pipeline.py --mode files --safe-train data/safe.txt ...")
print("\n" + "="*70)
