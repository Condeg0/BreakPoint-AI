import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import torch
from pathlib import Path

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

class Evaluator:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.plots_dir = run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def generate_report(self, y_true, y_prob, model=None, X_train=None, model_name="model", feature_names=None):
        print(f"\n>>> 📊 GENERATING EVALUATION REPORT FOR {model_name.upper()}...")

        # 1. Metrics
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        self._save_metrics(metrics)

        # 2. Plots
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_calibration_curve(y_true, y_prob)
        self._plot_metrics_summary(metrics)

        # 3. Explainability (SHAP)
        # We enforce strict checks to ensure we don't skip this silently
        if model is not None and X_train is not None:
            self._explain_model(model, X_train, model_name, feature_names)
        else:
            print("   ! Skipping SHAP: Model or Training Data missing.")

        print(f">>> Evaluation Complete. Reports saved to {self.run_dir}")

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob)
        }

    def _save_metrics(self, metrics):
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        report_str = f"""
========================================
        MODEL PERFORMANCE REPORT
========================================
AUC Score:   {metrics['auc']:.4f}
Accuracy:    {metrics['accuracy']:.4f}
Precision:   {metrics['precision']:.4f}
Recall:      {metrics['recall']:.4f}
F1 Score:    {metrics['f1']:.4f}
========================================
        """
        with open(self.run_dir / "report.txt", "w") as f:
            f.write(report_str)
        print(report_str)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Loss', 'Predicted Win'],
                    yticklabels=['Actual Loss', 'Actual Win'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png")
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "roc_curve.png")
        plt.close()

    def _plot_calibration_curve(self, y_true, y_prob):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='.', label='Model')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "calibration_curve.png")
        plt.close()

    def _plot_metrics_summary(self, metrics):
        names = list(metrics.keys())
        values = list(metrics.values())
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x=names, y=values, hue=names, legend=False, palette="viridis")
        plt.ylim(0, 1.0)
        plt.title("Key Performance Metrics")
        for i, p in enumerate(barplot.patches):
            height = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.2f}', ha="center")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_summary.png")
        plt.close()

    def _explain_model(self, model, X_train, model_name, feature_names=None):
        print(f"   - Starting SHAP calculation for {model_name}...")
        try:
            shap_vals_to_plot = None

            # 1. RANDOM FOREST / LOGREG
            if model_name in ["rf", "logreg"]:
                if hasattr(model, "model"): model = model.model

                # Unwrap CalibratedClassifierCV
                if hasattr(model, "calibrated_classifiers_"):
                    # Use the first estimator from calibration
                    base_est = model.calibrated_classifiers_[0].estimator if hasattr(model.calibrated_classifiers_[0], "estimator") else model.calibrated_classifiers_[0].base_estimator
                else:
                    base_est = model

                if model_name == "rf":
                    explainer = shap.TreeExplainer(base_est)
                    # Force check_additivity=False to prevent errors
                    shap_values = explainer.shap_values(X_train, check_additivity=False)
                else:
                    explainer = shap.LinearExplainer(base_est, X_train)
                    shap_values = explainer.shap_values(X_train)

                # Fix: TreeExplainer for binary classifier returns list [Class0, Class1].
                # We want Class 1 (Win).
                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1]
                # If it returned a 3D array (Interactions), sum over last dim to get main effects
                elif len(np.array(shap_values).shape) == 3:
                    print("     ! Detected Interaction Values. Flattening to Main Effects.")
                    shap_vals_to_plot = np.sum(shap_values, axis=-1)
                else:
                    shap_vals_to_plot = shap_values

            # 2. LSTM (Neural Network)
            elif model_name == "lstm":
                print("   - Configuring KernelExplainer for LSTM (this is slow)...")

                # Define wrapper to convert flat numpy array -> PyTorch tensors
                def lstm_predict(flat_data):
                    # Expect flat_data shape: (N, total_features)
                    device = next(model.parameters()).device
                    preds = []

                    # Infer dimensions from model config
                    seq_len = model.config.models.lstm.architecture.seq_len
                    input_dim = model.lstm.input_size

                    # Calculate split points
                    seq_a_end = seq_len * input_dim
                    seq_b_end = seq_a_end + (seq_len * input_dim)

                    for row in flat_data:
                        # Reshape back to (1, Seq, Feat)
                        s_a = row[:seq_a_end].reshape(1, seq_len, input_dim)
                        s_b = row[seq_a_end:seq_b_end].reshape(1, seq_len, input_dim)
                        ctx = row[seq_b_end:].reshape(1, -1)

                        t_a = torch.tensor(s_a, dtype=torch.float32).to(device)
                        t_b = torch.tensor(s_b, dtype=torch.float32).to(device)
                        t_c = torch.tensor(ctx, dtype=torch.float32).to(device)

                        with torch.no_grad():
                            logit = model(t_a, t_b, t_c)
                            prob = torch.sigmoid(logit).item()
                            preds.append(prob)

                    return np.array(preds)

                # Use a very small background sample for speed (e.g. 10-20 samples)
                # X_train passed here is the flattened numpy array
                explainer = shap.KernelExplainer(lstm_predict, X_train[:10])

                # Explain a small sample of test data (e.g. 20 samples)
                print("     Running KernelExplainer on 20 samples...")
                shap_values = explainer.shap_values(X_train[:20], silent=True)
                shap_vals_to_plot = shap_values

            # PLOT GENERATION
            if shap_vals_to_plot is not None:
                # Ensure dimensions match
                if feature_names and len(feature_names) != shap_vals_to_plot.shape[1]:
                    print(f"     ! Feature name mismatch: Names={len(feature_names)}, SHAP={shap_vals_to_plot.shape[1]}. Truncating names.")
                    feature_names = feature_names[:shap_vals_to_plot.shape[1]]

                plt.figure(figsize=(10, 8))
                # Force plot_type="dot" to standard beeswarm
                shap.summary_plot(shap_vals_to_plot, X_train if model_name != "lstm" else X_train[:20],
                                feature_names=feature_names, show=False, plot_type="dot", max_display=20)
                plt.title(f"SHAP Summary ({model_name})")
                plt.tight_layout()
                plt.savefig(self.plots_dir / "shap_summary.png")
                plt.close()
                print("   - SHAP Summary Plot saved successfully.")
            else:
                print("   ! SHAP values were None. Skipping plot.")

        except Exception as e:
            print(f"   ! Critical Error generating SHAP plots: {e}")
            import traceback
            traceback.print_exc()
