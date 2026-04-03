# BreakPoint AI: ATP Tennis Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Optuna](https://img.shields.io/badge/Optuna-Optimization-green)
![License](https://img.shields.io/badge/License-MIT-grey)

**BreakPoint AI** deploys a **Hybrid Siamese LSTM** architecture to model player momentum sequences. By enforcing strict **Time-Series Validation**, it eliminates look-ahead bias, guaranteeing institutional-grade signal extraction free from temporal leakage.

## 🚀 Impact & Architecture

* **Engineered a Siamese LSTM Pipeline:** Processed raw sequence histories (aces, faults, serve percentages) through twin LSTMs to generate latent "momentum embeddings," matching optimized tabular models without requiring manual feature smoothing.
* **Resolved Temporal Leakage:** Implemented a stateful inference API using strict `date < current_date` filtration to process rolling 10-match windows dynamically, ensuring the model never sees future stats in the history buffer.
* **Rich Feature Engineering:** Context features span surface-specific rolling win rates, serve efficiency ratios (break point save rate, first/second serve win %), log-transformed rank points differential, win/loss streak, rank momentum trend, and surface-specific head-to-head records — all computed with `shift(1)` leakage prevention.
* **Batch CLI Inference:** Loads frozen model weights locally and runs end-to-end predictions from a `.csv` of upcoming matches, outputting calibrated win probabilities.
* **Context Fusion:** Combines momentum embeddings with 23 match context features (Rank Points Diff, Surface, H2H, Serve Efficiency) in a dense fusion layer.
* **Institutional Evaluation:** Focuses on calibration (Reliability Diagrams) and SHAP values rather than just raw accuracy, mirroring financial risk modeling standards.

## 📊 Performance Benchmark

The pipeline uses a stacking ensemble of four base models. All results are evaluated on an unseen 2024 test set.

| Model | Test AUC | Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Stacking Ensemble** | **0.7168** | **65.35%** | XGBoost meta-learner over all base models |
| **XGBoost** | **0.7185** | **65.23%** | Strongest individual model |
| **Siamese LSTM** | **0.7140** | **65.50%** | Raw sequence history, no manual feature smoothing |
| **Random Forest** | **0.7112** | **65.00%** | Calibrated with sigmoid post-processing |
| **Logistic Regression** | **0.7086** | **65.05%** | Linear baseline |

> *In high-variance domains like ATP Tennis, an AUC of ~0.71–0.72 represents a meaningful statistical edge. The academic literature places the ceiling for purely statistical ATP prediction at approximately 0.75–0.77 AUC.*

## 🛠️ System Architecture

### 1. Directory Structure
```text
tennis-forecast/
├── configs/               # YAML configuration (hyperparams, features, splits)
├── src/
│   ├── data.py            # Dual-pipeline dataset (tabular vs. LSTM sequence mode)
│   ├── features.py        # Feature engineering (rolling, serve stats, streak, H2H)
│   ├── inference.py       # MetaLearnerPipeline (stateful inference)
│   ├── models/            # SiameseLSTM, XGBoost, RF, LogReg, Stacking
│   ├── training.py        # Training loop with early stopping
│   ├── tuning.py          # Optuna hyperparameter search
│   └── evaluation.py      # SHAP, calibration curves, ROC, metrics
├── cli_batch_predict.py   # CLI entrypoint for batch inference
└── main.py                # CLI entrypoint for training
```

### 2. Data Pipeline

**Ingestion:** Merges raw ATP match logs (1990–2024), filtered to exclude Davis Cup and Laver Cup.

**Temporal splits** (no leakage between splits):
- Train: up to 2022-12-31
- Validation: 2023
- Test: 2024 onward

**Feature Engineering** (`src/features.py`):

| Feature Group | Features | Method |
| :--- | :--- | :--- |
| Rolling form | Win rate, ace, DF, serve % (10-match window) | `shift(1)` → `rolling()` |
| Surface form | Win rate per surface | `shift(1)` → `rolling()` by (player, surface) |
| Serve efficiency | BP save rate, 1st/2nd serve win % | Derived ratio → `shift(1)` → `rolling()` |
| Rank quality | Log-transformed rank points diff | Pre-match ATP points (no leakage) |
| Momentum | Win/loss streak, rank trend | `shift(1)` on labels / rank |
| H2H | Overall and surface-specific win rate | `shift(1)` → `cumsum()` |
| Context | Rank diff, days since last match | Direct columns |

**Preprocessing:** `StandardScaler` and `OneHotEncoder` fitted on the training split only.

### 3. Model Diagram

The system treats a match as the collision of two player histories, fused with match context.

```mermaid
graph TD
    A[Player A History (10 × seq_features)] -->|Shared LSTM| E1[Momentum Embedding A]
    B[Player B History (10 × seq_features)] -->|Shared LSTM| E2[Momentum Embedding B]
    C[Match Context (23 features)] --> F[Fusion Layer]
    E1 --> F
    E2 --> F
    F -->|Dense 64 → Dense 32 → Dense 1| O[Win Probability]
```

## 💻 Usage

### 1. Installation

```bash
git clone https://github.com/condeg0/breakpoint-ai.git
cd breakpoint-ai
pip install -r requirements.txt
```

### 2. Train Base Models

```bash
python main.py --config configs/config.yaml
```

Set `pipeline.models_to_train` in `configs/config.yaml` to select which models to train:
```yaml
models_to_train: ["xgboost", "random_forest", "logistic_regression"]
# or: ["lstm"] for the Siamese LSTM
# or: ["stacking"] to train the meta-learner from a previous run's artifacts
```

### 3. Train the Stacking Ensemble

After training base models, point `stacking_base_artifact_dir` to the run's artifact folder and run:

```bash
# In config.yaml: models_to_train: ["stacking"]
#                 stacking_base_artifact_dir: "artifacts/<run_id>"
python main.py --config configs/config.yaml
```

### 4. CLI Batch Inference

Add upcoming matches to `data/inference/upcoming_matches.csv`, then run:

```bash
python cli_batch_predict.py --config configs/config.yaml --model stacking
```

Predictions are written to `data/inference/predictions.csv` with calibrated win probabilities for each match.

## 📈 Analysis & Visualizations

**1. Calibration (Reliability Diagram)**
When the model predicts a 70% win probability, the player wins ~70% of the time. This calibration property is critical for any expected value calculation in risk or betting applications.

**2. SHAP Values (Feature Importance)**
`rank_pts_diff` (log-transformed rank points differential) is the single strongest predictor (|correlation| = 0.338), surpassing rank ordinal difference (0.228). Surface-specific rolling win rate and first-serve win percentage are the next most important features.

**3. ROC Curve**
The ensemble achieves AUC ~0.717, demonstrating consistent separation between winning and losing outcomes across the 2024 test season.

---

⚖️ **Disclaimer:** This project is for educational and research purposes. The methodology emphasises calibration and risk-adjusted analysis over raw accuracy, aligning with institutional risk modeling standards. It is not financial advice.
