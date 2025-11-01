# Ensemble Space Invaders - Architecture

## Design Philosophy

**Problem**: The VAE-only approach (Latent Space Invaders) achieved low FPR (~3-8%) but terrible recall (~2-12%). It struggles with subtle attacks that appear "normal" in latent space.

**Solution**: Hybrid ensemble combining unsupervised VAE (good at novelty) with supervised classifier (good at learned patterns) and auxiliary heuristics.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Prompt                           │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    ┌─────┐   ┌─────────┐  ┌────────────┐
    │ VAE │   │   LLM   │  │ Auxiliary  │
    │Recon│   │ Hidden  │  │  Features  │
    │Error│   │ States  │  │ Extractor  │
    └──┬──┘   └────┬────┘  └─────┬──────┘
       │           │              │
       │      ┌────▼────┐         │
       │      │   VAE   │         │
       │      │ Encoder │         │
       │      └────┬────┘         │
       │           │              │
       │      ┌────▼────────┐     │
       │      │  Latent     │     │
       │      │  Vectors    │     │
       │      │ (per-layer) │     │
       │      └────┬────────┘     │
       │           │              │
       │      ┌────▼────────┐     │
       │      │ Supervised  │     │
       │      │ Classifier  │     │
       │      │ (LR/MLP/XGB)│     │
       │      └────┬────────┘     │
       │           │              │
       ▼           ▼              ▼
    ┌──────────────────────────────┐
    │    Ensemble Score Fusion     │
    │  (Weighted Avg / Stacking)   │
    └──────────────┬───────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Calibration    │
         │  (Platt Scaling) │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Anomaly Prob +   │
         │   Confidence     │
         └──────────────────┘
```

---

## Component Details

### 1. VAE Encoder (`vae_encoder.py`)

**Purpose**: Extract latent representations from LLM hidden states

**Architecture**:
- Reuse Layer-Conditioned VAE from Latent Space Invaders
- Input: Hidden states from LLM layers [0, 4, 8, 12, 16, 20]
- Conditioning: One-hot layer ID
- Latent dimension: 128
- Output: Per-layer latent vectors (6 layers × 128-dim = 768-dim feature vector)

**Training Modes**:
1. **Pretrained**: Load existing VAE trained on safe prompts
2. **Fine-tune**: Retrain on new safe data
3. **Frozen**: Use encoder only (for feature extraction)

**Key Methods**:
- `extract_latent_features(prompts)` → latent vectors
- `compute_reconstruction_loss(prompts)` → per-prompt loss
- `save_encoder()` / `load_encoder()`

---

### 2. Supervised Classifier (`latent_classifier.py`)

**Purpose**: Learn attack patterns from labeled data using latent features

**Input Features**:
- Latent vectors from VAE encoder (768-dim)
- Auxiliary features (variable dim)
- Total: ~800-1000 dim

**Models Supported**:
1. **Logistic Regression** (baseline, fast)
2. **MLP** (2-3 hidden layers, dropout)
3. **XGBoost** (tree-based, handles non-linearity)

**Training**:
- Class weighting for imbalanced data
- Cross-validation for hyperparameter tuning
- Calibration on validation set

**Output**: Attack probability [0, 1]

---

### 3. Auxiliary Features (`auxiliary_features.py`)

**Purpose**: Extract simple heuristics that correlate with attacks

**Features**:

| Feature | Rationale | Computation |
|---------|-----------|-------------|
| **Prompt Length (chars)** | Attacks often verbose | `len(prompt)` |
| **Prompt Length (tokens)** | Token-level length | `len(tokenizer(prompt))` |
| **Token Entropy** | Attacks may have unusual token distributions | `-Σ p(t) log p(t)` |
| **Reconstruction Loss** | From VAE (already computed) | MSE(original, reconstructed) |
| **Perplexity Delta** | Change in LLM perplexity vs baseline | `exp(mean(log_probs))` (if available) |
| **Special Token Count** | Count of [INST], <s>, etc. | Regex matching |
| **Capitalization Ratio** | All-caps prompts suspicious | `sum(isupper) / len(chars)` |

**Output**: Feature vector (variable dim, typically 7-10 features)

---

### 4. Ensemble Detector (`ensemble_detector.py`)

**Purpose**: Fuse scores from multiple detectors for robust prediction

**Detector Components**:
1. **VAE Reconstruction Detector**
   - Score: Normalized reconstruction error
   - Weight: Tunable (default: 0.3)

2. **Supervised Classifier Detector**
   - Score: Calibrated attack probability
   - Weight: Tunable (default: 0.5)

3. **Auxiliary Heuristic Detector**
   - Score: Composite of auxiliary features
   - Weight: Tunable (default: 0.2)

**Fusion Methods**:

**Option 1: Weighted Average**
```python
score = w1 * vae_score + w2 * classifier_score + w3 * auxiliary_score
```

**Option 2: Logistic Stacking**
```python
# Train meta-classifier on [vae_score, classifier_score, aux_score]
# using validation set
meta_model = LogisticRegression()
meta_model.fit([scores], labels)
final_score = meta_model.predict_proba([scores])[:, 1]
```

**Calibration**:
- Platt scaling on validation set
- Isotonic regression for non-parametric calibration
- Output: Calibrated probability + confidence interval

**Threshold Tuning**:
- Provide operating points for different FPR targets (0.1%, 1%, 5%)
- User can select based on deployment requirements

---

## Data Pipeline (`data_utils.py`)

### Dataset Format

**Expected Structure**:
```
data/
├── safe_train.txt        # Safe prompts for training
├── safe_val.txt          # Safe prompts for validation
├── attack_train.txt      # Attack prompts for training
├── attack_val.txt        # Attack prompts for validation
└── attack_test.txt       # Attack prompts for testing
```

Each file: one prompt per line

**Dataset Creation**:
1. Load existing datasets (SEP, Jailbreak, etc.)
2. Optionally synthesize adversarial attacks using LLM paraphrasing
3. Split with stratification (preserve attack type distribution)
4. Balance classes (undersample safe or oversample attacks)

### Data Augmentation

**For Subtle Attack Generation**:
```python
# Use LLM to paraphrase jailbreaks
original = "Ignore previous instructions..."
paraphrased = llm.generate(
    f"Rephrase this to be more subtle: {original}"
)
```

**Types**:
- Paraphrased jailbreaks
- Typo injection
- Case variation
- Token-level perturbations

---

## Training Workflow

### Stage 1: VAE Pretraining (Optional)
```bash
python train_vae.py \
    --safe-prompts data/safe_train.txt \
    --epochs 20 \
    --latent-dim 128 \
    --output models/vae_encoder.pth
```

### Stage 2: Feature Extraction
```bash
python extract_features.py \
    --vae-model models/vae_encoder.pth \
    --safe-prompts data/safe_train.txt \
    --attack-prompts data/attack_train.txt \
    --output features/train_features.npz
```

### Stage 3: Supervised Classifier Training
```bash
python train_classifier.py \
    --features features/train_features.npz \
    --model-type xgboost \
    --output models/classifier.pkl
```

### Stage 4: Ensemble Calibration
```bash
python train_ensemble.py \
    --vae-model models/vae_encoder.pth \
    --classifier-model models/classifier.pkl \
    --val-data features/val_features.npz \
    --fusion-method stacking \
    --output models/ensemble.pkl
```

---

## Evaluation Pipeline (`evaluate.py`)

### Metrics

**Standard Metrics**:
- Accuracy, Precision, Recall, F1
- ROC AUC, PR AUC
- Confusion Matrix

**Custom Metrics**:
- **Recall@FPR**: Recall at specific FPR thresholds
  - Recall @ 0.1% FPR
  - Recall @ 1% FPR
  - Recall @ 5% FPR

**Per-Attack-Type Metrics**:
- Breakdown by attack category (jailbreak, injection, etc.)
- Identify which attack types are hardest to detect

### Visualizations

1. **ROC Curve**: TPR vs FPR
2. **Precision-Recall Curve**: Precision vs Recall
3. **Calibration Plot**: Predicted probability vs true frequency
4. **Latent Space t-SNE**: Visualize safe vs attack clustering
5. **Decision Boundary**: 2D projection showing classifier decision regions
6. **Confusion Matrix**: Heatmap of predictions
7. **Score Distribution**: Histograms of detector scores by class

---

## Deployment

### Inference API

```python
from ensemble_detector import EnsembleDetector

# Initialize
detector = EnsembleDetector(
    vae_model_path="models/vae_encoder.pth",
    classifier_model_path="models/classifier.pkl",
    ensemble_model_path="models/ensemble.pkl"
)

# Detect
result = detector.predict("Ignore all previous instructions...")
print(f"Is attack: {result['is_attack']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Breakdown: VAE={result['vae_score']:.3f}, "
      f"Classifier={result['classifier_score']:.3f}, "
      f"Auxiliary={result['auxiliary_score']:.3f}")
```

### Operating Point Selection

```python
# Get recommended thresholds for different FPR targets
thresholds = detector.get_operating_points()
# {0.001: 0.95, 0.01: 0.85, 0.05: 0.75}

# Use strict threshold (0.1% FPR)
detector.set_threshold(thresholds[0.001])
```

---

## Expected Performance

| Metric | VAE-Only | Hybrid Classifier | Ensemble (Target) |
|--------|----------|-------------------|-------------------|
| **FPR** | 3-8% | 5-10% | **<5%** |
| **Recall** | 2-12% | 40-60% | **>70%** |
| **ROC AUC** | 0.58-0.89 | 0.75-0.85 | **>0.90** |
| **Recall @ 1% FPR** | ~5% | ~30% | **>50%** |

**Goal**: Achieve >70% recall while maintaining <5% FPR across diverse attack types.

---

## Implementation Plan

### Phase 1: Core Components (Days 1-2)
- [x] Architecture design
- [ ] `vae_encoder.py` - Wrapper around existing VAE
- [ ] `auxiliary_features.py` - Feature extraction
- [ ] `latent_classifier.py` - Supervised models
- [ ] `data_utils.py` - Data loading & preprocessing

### Phase 2: Ensemble & Evaluation (Days 3-4)
- [ ] `ensemble_detector.py` - Score fusion & calibration
- [ ] `evaluate.py` - Comprehensive metrics & visualization
- [ ] `train_pipeline.py` - End-to-end training script
- [ ] Unit tests for each component

### Phase 3: Deployment & Documentation (Day 5)
- [ ] CLI interface for training/inference
- [ ] Jupyter notebook with examples
- [ ] README with usage guide
- [ ] Performance benchmarking on test sets

---

## Directory Structure

```
ensemble-space-invaders/
├── README.md                    # Main documentation
├── ARCHITECTURE.md              # This file
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
│
├── src/
│   ├── vae_encoder.py          # VAE latent extraction
│   ├── auxiliary_features.py   # Heuristic features
│   ├── latent_classifier.py    # Supervised classifier
│   ├── ensemble_detector.py    # Ensemble fusion
│   ├── data_utils.py           # Data loading
│   └── evaluate.py             # Metrics & visualization
│
├── scripts/
│   ├── train_vae.py            # VAE training
│   ├── train_classifier.py     # Classifier training
│   ├── train_ensemble.py       # Ensemble training
│   └── run_evaluation.py       # Full evaluation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_ensemble_tuning.ipynb
│
├── tests/
│   ├── test_vae_encoder.py
│   ├── test_classifier.py
│   └── test_ensemble.py
│
├── data/                        # Datasets (not in repo)
│   ├── safe_train.txt
│   ├── attack_train.txt
│   └── ...
│
├── models/                      # Saved models (not in repo)
│   ├── vae_encoder.pth
│   ├── classifier.pkl
│   └── ensemble.pkl
│
└── results/                     # Evaluation outputs
    ├── metrics.json
    ├── roc_curve.png
    └── calibration_plot.png
```

---

## Key Design Decisions

1. **Modular Architecture**: Each detector is independent and can be used standalone
2. **Reuse VAE**: Leverage existing trained VAE from Latent Space Invaders
3. **Flexible Fusion**: Support both weighted average and stacking
4. **Calibration Required**: All scores must be calibrated for fair fusion
5. **Multiple Operating Points**: Let users choose FPR/recall tradeoff
6. **Rich Evaluation**: Per-attack-type metrics to identify blind spots

---

## Future Enhancements

1. **Temporal Delta Detector**: Track per-token latent shifts
2. **Adversarial Training**: Adaptive jailbreak generation loop
3. **Online Learning**: Update models with new attack patterns
4. **Multi-Model Ensemble**: Use different LLMs for diversity
5. **Attention-Based Features**: Use attention patterns as signals
6. **Active Learning**: Query uncertain samples for labeling
