"""
Evaluation utilities for regression and classification models.

WHY A SEPARATE MODULE:
Evaluation code is called from train.py, register_best.py, and the
monitoring module. Keeping it here avoids duplication and ensures
consistent metric definitions across all three.

Regression metrics: RMSE, MAE, R2
Classification metrics: accuracy, precision (macro), recall (macro),
                        macro-F1, confusion matrix

All plots are saved to reports/figures/ using matplotlib (no display
required — suitable for headless servers and CI).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR
from src.ml.features import FEATURE_COLS, TARGET_REG, TARGET_CLF

# Suppress irrelevant warnings from sklearn metrics
warnings.filterwarnings("ignore", category=UserWarning)


# ── Baselines ─────────────────────────────────────────────────────────────────

def persistence_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Persistence model: predict tomorrow = today's AQI.
    This is the simplest meaningful baseline — AQI has day-to-day autocorrelation,
    so this is non-trivial to beat.
    """
    # aqi_lag1 is yesterday's AQI — that's what we'd know at prediction time
    # but to predict next-day AQI for each row, we use the current day's AQI
    return df["aqi_value"].fillna(method="ffill").values


def moving_average_baseline(
    df: pd.DataFrame, train_df: pd.DataFrame, window: int = 7
) -> np.ndarray:
    """
    7-day moving average baseline.
    Concatenate train_df (as context) + df, compute rolling mean, then
    return only the part corresponding to df — no leakage because we
    never use future data.
    """
    combined = pd.concat([train_df[["aqi_value"]], df[["aqi_value"]]], ignore_index=True)
    # shift(1) so each prediction uses only past data
    rolling = combined["aqi_value"].shift(1).rolling(window=window, min_periods=1).mean()
    return rolling.iloc[-len(df):].fillna(method="ffill").values


# ── Regression metrics ─────────────────────────────────────────────────────────

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute RMSE, MAE, and R² for regression predictions.

    Args:
        y_true: actual AQI values
        y_pred: predicted AQI values
        label:  display name for logging

    Returns:
        dict with keys: rmse, mae, r2
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Remove NaN pairs (can occur at split boundaries)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"[eval] {label:30s}  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R²={r2:+.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ── Classification metrics ─────────────────────────────────────────────────────

def evaluate_classification(
    y_true,
    y_pred,
    label: str = "",
) -> dict:
    """
    Compute accuracy, precision, recall, macro-F1, and confusion matrix.

    Uses macro averaging (equal weight per class) because AQI categories
    are naturally imbalanced (most days are Moderate, few are Hazardous).
    Macro F1 penalises the model equally for failing on any category.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()

    print(
        f"[eval] {label:30s}  Acc={acc:.3f}  Prec={prec:.3f}  "
        f"Rec={rec:.3f}  F1={f1:.3f}"
    )
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "macro_f1": f1,
        "confusion_matrix": cm,
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def save_comparison_table(all_results: dict, baselines: dict):
    """
    Save a CSV and Markdown comparison table of all models vs baselines.
    Format: one row per model, columns for val and test metrics.

    WHY SAVE BOTH CSV AND MD:
    CSV → can be read by Streamlit dashboard and pandas for further analysis.
    Markdown → human-readable in GitHub and the Part 2 report.
    """
    rows = []

    # Baselines (regression only, for reference)
    for bl_name, bl_metrics in baselines.items():
        rows.append({
            "type": "regression", "model": bl_name.replace("_", " "), "split": "baseline",
            "rmse": bl_metrics.get("rmse"), "mae": bl_metrics.get("mae"),
            "r2": bl_metrics.get("r2"), "accuracy": None, "macro_f1": None,
        })

    # Regression models
    for model_name, result in all_results["regression"].items():
        for split_name, metrics in [("val", result["val"]), ("test", result["test"])]:
            rows.append({
                "type": "regression", "model": model_name, "split": split_name,
                "rmse": metrics.get("rmse"), "mae": metrics.get("mae"),
                "r2": metrics.get("r2"), "accuracy": None, "macro_f1": None,
            })

    # Classification models
    for model_name, result in all_results["classification"].items():
        for split_name, metrics in [("val", result["val"]), ("test", result["test"])]:
            rows.append({
                "type": "classification", "model": model_name, "split": split_name,
                "rmse": None, "mae": None, "r2": None,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
            })

    df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[eval] Saved model_comparison.csv → {csv_path}")

    # Markdown version
    md_path = REPORTS_DIR / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write("## Regression Models\n\n")
        reg_df = df[df["type"] == "regression"][["model", "split", "rmse", "mae", "r2"]]
        f.write(reg_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Classification Models\n\n")
        clf_df = df[df["type"] == "classification"][["model", "split", "accuracy", "macro_f1"]]
        f.write(clf_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n> Metrics computed on chronological val/test splits only. "
                "No data leakage. Random seed = 42.\n")
    print(f"[eval] Saved model_comparison.md  → {md_path}")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_regression_plots(reg_results, val_df, test_df, X_val, X_test, y_val, y_test):
    """
    For each regression model:
    - Predicted vs Actual scatter (test split)
    - Residuals over time (test split)
    - Feature importance (if available)
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — no display required
    import matplotlib.pyplot as plt

    for model_name, result in reg_results.items():
        model = result["model"]
        y_pred_test = model.predict(X_test)

        # ── Predicted vs Actual ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=20)
        mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="Perfect prediction")
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        ax.set_title(f"{model_name} — Predicted vs Actual (Test)")
        ax.legend()
        fig.tight_layout()
        path = FIGURES_DIR / f"reg_{model_name.lower()}_pred_vs_actual.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)

        # ── Residuals over time ────────────────────────────────────────────
        residuals = y_test - y_pred_test
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(test_df["date"].values, residuals, lw=1)
        ax.axhline(0, color="red", linestyle="--", lw=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title(f"{model_name} — Residuals over Time (Test)")
        fig.tight_layout()
        path = FIGURES_DIR / f"reg_{model_name.lower()}_residuals.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)

        # ── Feature importance (tree models only) ─────────────────────────
        inner_model = model.named_steps.get("model")
        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
            sorted_idx = np.argsort(importances)[-20:]  # top 20
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh([FEATURE_COLS[i] for i in sorted_idx],
                    importances[sorted_idx], color="steelblue")
            ax.set_title(f"{model_name} — Feature Importances (Top 20)")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            path = FIGURES_DIR / f"reg_{model_name.lower()}_feature_importance.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)


def save_classification_plots(clf_results, X_val, X_test, y_val, y_test, le):
    """Save confusion matrix plot for each classification model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    for model_name, result in clf_results.items():
        model = result["model"]
        preds = model.predict(X_test)
        classes = le.classes_

        cm = confusion_matrix(y_test, preds, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(8, 7))
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"{model_name} — Confusion Matrix (Test)")
        fig.tight_layout()
        path = FIGURES_DIR / f"clf_{model_name.lower()}_confusion_matrix.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
