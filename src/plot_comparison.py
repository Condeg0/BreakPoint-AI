import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import joblib
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from config import ProjectConfig
from models.nn import SiameseLSTM
from data import Preprocessor, TennisDataset

# Load Configs
lstm_cfg = ProjectConfig.load("configs/config_lstm.yaml")
rf_cfg = ProjectConfig.load("configs/config_rf.yaml")

# Load Artifacts (Update paths to your actual best run folders)
# Example: artifacts/lstm/20260130_205011/
LSTM_PATH = "artifacts/lstm/LATEST_RUN_ID" 
RF_PATH = "artifacts/rf/LATEST_RUN_ID"

# ... (Load Data logic similar to main.py, but for Test set only) ...
# [Simulated Plotting Code for brevity - run this logic in your notebook]

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

# Plot RF
# prob_true_rf, prob_pred_rf = calibration_curve(y_test, rf_probs, n_bins=10)
# plt.plot(prob_pred_rf, prob_true_rf, "s-", label="Random Forest (Baseline)")

# Plot LSTM
# prob_true_lstm, prob_pred_lstm = calibration_curve(y_test, lstm_probs, n_bins=10)
# plt.plot(prob_pred_lstm, prob_true_lstm, "o-", label="Siamese LSTM (Ours)")

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Win Rate")
plt.title("Model reliability: Deep Learning vs. Baseline")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("reports/final_comparison.png")