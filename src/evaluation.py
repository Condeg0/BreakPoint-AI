import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from src.logger import get_logger

logger = get_logger(__name__)

class Evaluator:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir: Path = run_dir
        self.plots_dir: Path = run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def generate_report(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        model: Optional[Any] = None, 
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None, 
        model_name: str = "model", 
        feature_names: Optional[List[str]] = None
    ) -> None:
        logger.info(f"\n>>> 📊 GENERATING EVALUATION REPORT FOR {model_name.upper()}...")

        y_pred: np.ndarray = (y_prob >= 0.5).astype(int)
        metrics: Dict[str, float] = self._calculate_metrics(y_true, y_pred, y_prob)
        self._save_metrics(metrics)

        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_calibration_curve(y_true, y_prob)
        self._plot_metrics_summary(metrics)

        if model is not None and X_train is not None:
            self._explain_model(model, X_train, model_name, feature_names)
        else:
            logger.warning("   ! Skipping SHAP: Model or Training Data missing.")

        logger.info(f">>> Evaluation Complete. Reports saved to {self.run_dir}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, y_prob))
        }

    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        report_str: str = f"""
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
        logger.info(report_str)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm: np.ndarray = confusion_matrix(y_true, y_pred)
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

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        fpr: np.ndarray
        tpr: np.ndarray
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc: float = auc(fpr, tpr)
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

    def _plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        prob_true: np.ndarray
        prob_pred: np.ndarray
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

    def _plot_metrics_summary(self, metrics: Dict[str, float]) -> None:
        names: List[str] = list(metrics.keys())
        values: List[float] = list(metrics.values())
        plt.figure(figsize=(10, 6))
        barplot: plt.Axes = sns.barplot(x=names, y=values, hue=names, legend=False, palette="viridis")
        plt.ylim(0, 1.0)
        plt.title("Key Performance Metrics")
        for p in barplot.patches:
            height: float = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.2f}', ha="center")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_summary.png")
        plt.close()

    def _explain_model(self, model: Any, X_train: Union[pd.DataFrame, np.ndarray], model_name: str, feature_names: Optional[List[str]]) -> None:
        logger.info(f"   - Starting SHAP calculation for {model_name}...")
        try:
            shap_vals_to_plot: Optional[np.ndarray] = None
            explainer: Union[shap.TreeExplainer, shap.LinearExplainer, shap.KernelExplainer]

            if model_name in ["rf", "logreg", "random_forest", "logistic_regression", "xgboost", "stacking"]:
                base_model: Any = model.model if hasattr(model, "model") else model
                
                if hasattr(base_model, "calibrated_classifiers_"):
                    base_est: Any = base_model.calibrated_classifiers_[0].base_estimator
                else:
                    base_est = base_model

                shap_values: Union[np.ndarray, List[np.ndarray]]
                if model_name in ["rf", "random_forest", "xgboost"]:
                    explainer = shap.TreeExplainer(base_est)
                    shap_values = explainer.shap_values(X_train, check_additivity=False)
                else:
                    explainer = shap.LinearExplainer(base_est, X_train)
                    shap_values = explainer.shap_values(X_train)

                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1]
                elif len(np.array(shap_values).shape) == 3:
                    logger.info("     ! Detected Interaction Values. Flattening to Main Effects.")
                    shap_vals_to_plot = np.sum(shap_values, axis=-1)
                else:
                    shap_vals_to_plot = shap_values

            elif model_name == "lstm":
                logger.info("   - Configuring KernelExplainer for LSTM (this is slow)...")

                def lstm_predict(flat_data: np.ndarray) -> np.ndarray:
                    device: torch.device = next(model.parameters()).device
                    preds: List[float] = []

                    seq_len: int = model.config.models.lstm.architecture.seq_len
                    input_dim: int = model.lstm.input_size
                    seq_a_end: int = seq_len * input_dim
                    seq_b_end: int = seq_a_end + (seq_len * input_dim)

                    for row in flat_data:
                        s_a: np.ndarray = row[:seq_a_end].reshape(1, seq_len, input_dim)
                        s_b: np.ndarray = row[seq_a_end:seq_b_end].reshape(1, seq_len, input_dim)
                        ctx: np.ndarray = row[seq_b_end:].reshape(1, -1)

                        t_a: torch.Tensor = torch.tensor(s_a, dtype=torch.float32).to(device)
                        t_b: torch.Tensor = torch.tensor(s_b, dtype=torch.float32).to(device)
                        t_c: torch.Tensor = torch.tensor(ctx, dtype=torch.float32).to(device)

                        with torch.no_grad():
                            logit: torch.Tensor = model(t_a, t_b, t_c)
                            preds.append(torch.sigmoid(logit).item())

                    return np.array(preds)

                explainer = shap.KernelExplainer(lstm_predict, X_train[:10])
                logger.info("     Running KernelExplainer on 20 samples...")
                shap_values_lstm: np.ndarray = explainer.shap_values(X_train[:20], silent=True)
                shap_vals_to_plot = shap_values_lstm

            if shap_vals_to_plot is not None:
                if feature_names and len(feature_names) != shap_vals_to_plot.shape[1]:
                    logger.warning(f"Feature name mismatch: Names={len(feature_names)}, SHAP={shap_vals_to_plot.shape[1]}. Truncating.")
                    feature_names = feature_names[:shap_vals_to_plot.shape[1]]

                X_plot: Union[pd.DataFrame, np.ndarray] = X_train if not isinstance(X_train, np.ndarray) else (X_train if model_name != "lstm" else X_train[:20])
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_vals_to_plot, X_plot, feature_names=feature_names, show=False, plot_type="dot", max_display=20)
                plt.title(f"SHAP Summary ({model_name})")
                plt.tight_layout()
                plt.savefig(self.plots_dir / "shap_summary.png")
                plt.close()
                logger.info("   - SHAP Summary Plot saved successfully.")
            else:
                logger.warning("   ! SHAP values were None. Skipping plot.")

        except Exception as e:
            logger.error(f"   ! Critical Error generating SHAP plots: {e}", exc_info=True)
