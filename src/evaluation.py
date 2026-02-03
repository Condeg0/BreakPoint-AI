import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
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
        # Set professional style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def generate_report(self, y_true, y_prob, model=None, X_train=None, model_name="model"):
        print(f"\n>>> 📊 GENERATING EVALUATION REPORT FOR {model_name.upper()}...")

        # 1. Calculate Scalar Metrics
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        self._save_metrics(metrics)

        # 2. Performance Plots
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_calibration_curve(y_true, y_prob)
        self._plot_metrics_summary(metrics)

        # 3. Explainability Plots (Baselines Only)
        if model is not None and X_train is not None:
            self._explain_model(model, X_train, model_name)

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
        # --- FIX: Removed deprecation warning by assigning x to hue ---
        barplot = sns.barplot(x=names, y=values, hue=names, legend=False, palette="viridis")
        plt.ylim(0, 1.0)
        plt.title("Key Performance Metrics")

        for i, p in enumerate(barplot.patches):
            height = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.01,
                         f'{height:.2f}', ha="center")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_summary.png")
        plt.close()

    def _explain_model(self, model, X_train, model_name):
        # A. Feature Importance
        try:
            if hasattr(model, "model"):
                inner_model = model.model
            else:
                inner_model = model

            importances = None
            if hasattr(inner_model, "feature_importances_"):
                importances = inner_model.feature_importances_
            elif hasattr(inner_model, "coef_"):
                importances = inner_model.coef_[0]

            if importances is not None:
                if hasattr(X_train, "columns"):
                    features = X_train.columns
                else:
                    features = [f"Feature {i}" for i in range(len(importances))]

                df_imp = pd.DataFrame({'feature': features, 'importance': importances})
                df_imp = df_imp.sort_values('importance', key=abs, ascending=False).head(20)

                plt.figure(figsize=(10, 8))
                # --- FIX: Updated barplot arguments for future proofing ---
                sns.barplot(data=df_imp, y='feature', x='importance', hue='feature', legend=False, palette="mako")
                plt.title(f'Top 20 Feature Importances ({model_name})')
                plt.tight_layout()
                plt.savefig(self.plots_dir / "feature_importance.png")
                plt.close()
                print("   - Feature Importance Plot saved.")
        except Exception as e:
            print(f"   ! Could not generate Feature Importance: {e}")

        # B. SHAP Values
        if model_name in ["rf", "logreg"]:
            try:
                print("   - Calculating SHAP values (this may take a moment)...")

                if hasattr(model, "model"):
                    inner_model = model.model
                else:
                    inner_model = model

                # --- FIX: ROBUST SKLEARN VERSION HANDLING ---
                # Check for CalibratedClassifierCV wrapper
                if hasattr(inner_model, "calibrated_classifiers_"):
                    # Get the first sub-estimator
                    calibrated_clf = inner_model.calibrated_classifiers_[0]

                    # Newer Sklearn uses 'estimator', Older uses 'base_estimator'
                    if hasattr(calibrated_clf, "estimator"):
                        explainer_model = calibrated_clf.estimator
                    elif hasattr(calibrated_clf, "base_estimator"):
                         explainer_model = calibrated_clf.base_estimator
                    else:
                        raise AttributeError("Could not find 'estimator' or 'base_estimator' in CalibratedClassifier")
                else:
                    explainer_model = inner_model

                if model_name == "rf":
                    explainer = shap.TreeExplainer(explainer_model)
                else:
                    explainer = shap.LinearExplainer(explainer_model, X_train)

                sample_data = X_train.sample(min(500, len(X_train)))
                shap_values = explainer.shap_values(sample_data)

                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1]
                else:
                    shap_vals_to_plot = shap_values

                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_vals_to_plot, sample_data, show=False)
                plt.title(f"SHAP Summary ({model_name})")
                plt.tight_layout()
                plt.savefig(self.plots_dir / "shap_summary.png")
                plt.close()
                print("   - SHAP Summary Plot saved.")

            except Exception as e:
                print(f"   ! Could not generate SHAP plots: {e}")
