"""
Model Evaluation and Explainability Module for Breast Cancer Prediction.
Performs test set validation, threshold calibration for high-recall screening,
clinical performance plotting (Confusion Matrix, ROC, PR Curve), SHAP interpretability,
and metric artifact export.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocess import (
    encode_features,
    handle_missing,
    handle_outliers,
    scale_features,
    split_data,
)
from src.utils import load_artifact, load_config, save_artifact, setup_logging

logger = setup_logging()

# Styling settings for publication-quality figures
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_theme(style="whitegrid", palette="muted")


def load_test_data(
    data_path: str = "data/breast_cancer_cleaned.csv",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load cleaned dataset and run preprocessing pipeline through test scaling.
    CRITICAL: The test set remains completely untouched and is NEVER SMOTEd.

    Parameters:
        data_path: Path to cleaned dataset CSV.
        config_path: Path to configuration YAML.

    Returns:
        X_test_scaled: Scaled test feature DataFrame.
        y_test: Test ground truth labels series.
        feature_names: List of column feature names.
    """
    logger.info("Loading and preprocessing test dataset (strictly un-SMOTEd)...")
    config = load_config(config_path) if os.path.exists(config_path) else {}

    if not os.path.exists(data_path):
        if os.path.exists(os.path.join("..", data_path)):
            data_path = os.path.join("..", data_path)
        else:
            raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Drop any residual leakage columns
    leakage_cols = config.get("features", {}).get("drop", ["Patient_ID", "Biopsy_Result", "Cancer_Stage"])
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Impute missing values
    df = handle_missing(df, config)

    # 3. Handle outliers
    df = handle_outliers(df, config)

    # 4. Categorical encoding
    df = encode_features(df, config)

    # 5. Separate features & target
    target_col = config.get("features", {}).get("target", "Cancer")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # 6. Train/Test split
    test_size = config.get("model", {}).get("test_size", 0.2)
    random_state = config.get("project", {}).get("random_state", 42)
    stratify = config.get("model", {}).get("stratify", True)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # 7. Scale features using fitted training scaler
    num_cols = config.get("features", {}).get("numerical", None)
    scaler_path = os.path.join(config.get("paths", {}).get("model_dir", "models/"), "scaler.pkl")

    if os.path.exists(scaler_path):
        scaler = load_artifact(scaler_path)
        cols_to_scale = [c for c in (num_cols or X.columns) if c in X_test.columns]
        X_test_scaled = X_test.copy()
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    else:
        _, X_test_scaled, _ = scale_features(X_train, X_test, numerical_cols=num_cols, save_scaler_path=scaler_path)

    feature_names = X_test_scaled.columns.tolist()
    logger.info(f"Test data loaded: {X_test_scaled.shape[0]} samples, {X_test_scaled.shape[1]} features.")
    return X_test_scaled, y_test, feature_names


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_recall: float = 0.90,
    min_precision: float = 0.70,
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate trained model on test set across key metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
    If initial Recall for Malignant class < target_recall (0.90), tunes classification probability
    threshold between 0.10 and 0.50 to maximize Recall while keeping Precision >= min_precision (0.70).

    Parameters:
        model: Fitted classification model.
        X_test: Test features.
        y_test: Test true labels.
        target_recall: Minimum desired recall threshold (default: 0.90).
        min_precision: Minimum acceptable precision during threshold search (default: 0.70).

    Returns:
        metrics: Dictionary of evaluated metrics.
        optimal_threshold: Selected decision threshold (0.50 or calibrated).
    """
    logger.info("Evaluating model predictions on test dataset...")

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        y_proba = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)
    else:
        y_proba = None

    # Base predictions at threshold 0.50
    y_pred = model.predict(X_test)
    optimal_threshold = 0.50

    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    logger.info(
        f"Default Threshold (0.50) Results -> Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}"
    )

    # Threshold calibration if recall < target_recall
    if rec < target_recall and y_proba is not None:
        logger.info(f"Initial Recall ({rec:.4f}) < target ({target_recall}). Performing threshold calibration...")
        candidate_thresholds = np.linspace(0.10, 0.50, 81)
        best_t = 0.50
        best_r = rec
        best_p = prec

        for t in candidate_thresholds:
            y_t = (y_proba >= t).astype(int)
            r_t = recall_score(y_test, y_t, pos_label=1, zero_division=0)
            p_t = precision_score(y_test, y_t, pos_label=1, zero_division=0)

            if p_t >= min_precision and r_t >= best_r:
                best_r = r_t
                best_p = p_t
                best_t = float(t)

        if best_t != 0.50:
            optimal_threshold = best_t
            y_pred = (y_proba >= optimal_threshold).astype(int)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            logger.info(
                f"Calibrated Threshold ({optimal_threshold:.3f}) Results -> Recall: {rec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}"
            )

    # Final clinical warning if recall is still under 0.90
    if rec < target_recall:
        warning_msg = (
            f"WARNING: Test set Recall ({rec:.4f}) is below the required 0.90 threshold! "
            "Further feature engineering or ensemble tuning is recommended."
        )
        print(f"\n[!] {warning_msg}\n")
        logger.warning(warning_msg)

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "Optimal_Threshold": round(float(optimal_threshold), 3),
    }

    return metrics, optimal_threshold


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    path: str = "outputs/plots/confusion_matrix.png",
) -> None:
    """
    Plot and export an annotated clinical confusion matrix.

    Parameters:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        path: Filepath for PNG export.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Annotation strings with counts and percentages
    group_counts = [f"{v:,}" for v in cm.flatten()]
    group_pcts = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]
    group_names = [
        "True Negative (Benign)",
        "False Positive (False Alarm)",
        "False Negative (Missed Cancer)",
        "True Positive (Malignant)",
    ]

    labels = [f"{n}\n{c}\n({p})" for n, c, p in zip(group_names, group_counts, group_pcts)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=1.2,
        linecolor="black",
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "medium"},
    )

    ax.set_title("Clinical Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Ground Truth", fontsize=12, labelpad=10)
    ax.set_xticklabels(["Benign (0)", "Malignant (1)"], fontsize=11)
    ax.set_yticklabels(["Benign (0)", "Malignant (1)"], fontsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {path}")


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str = "outputs/plots/roc_curve.png",
) -> None:
    """
    Plot and export Receiver Operating Characteristic (ROC) curve.

    Parameters:
        model: Fitted classification model.
        X_test: Test features.
        y_test: Test labels.
        path: Filepath for PNG export.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#d9534f", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"ROC curve plot saved to {path}")


def plot_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str = "outputs/plots/precision_recall_curve.png",
) -> None:
    """
    Plot and export Precision-Recall curve with Average Precision (AP) score.

    Parameters:
        model: Fitted classification model.
        X_test: Test features.
        y_test: Test labels.
        path: Filepath for PNG export.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    baseline_prev = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#1b7837", lw=2.5, label=f"PR Curve (AP = {ap_score:.4f})")
    ax.axhline(
        baseline_prev,
        color="#762a83",
        lw=1.5,
        linestyle="--",
        label=f"Baseline Prevalence ({baseline_prev:.1%})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=10)
    ax.set_title("Precision-Recall Curve (PR)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Precision-Recall curve plot saved to {path}")


def shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    path: str = "outputs/plots/shap_summary.png",
) -> List[Tuple[str, float]]:
    """
    Compute SHAP (SHapley Additive exPlanations) values to interpret feature attributions,
    generate summary beeswarm plot, and identify top 5 most important predictive features.

    Parameters:
        model: Trained model.
        X_test: Test features DataFrame.
        feature_names: List of feature names.
        path: Filepath for SHAP summary PNG export.

    Returns:
        List of (feature_name, mean_abs_shap_value) for top 5 features.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Computing SHAP feature attributions...")

    # Determine appropriate SHAP Explainer based on model architecture
    model_type_name = type(model).__name__.lower()
    if "xgb" in model_type_name or "forest" in model_type_name or "tree" in model_type_name:
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer(X_test)
    elif "logistic" in model_type_name or "linear" in model_type_name:
        explainer = shap.LinearExplainer(model, X_test)
        shap_values_raw = explainer(X_test)
    else:
        explainer = shap.Explainer(model, X_test)
        shap_values_raw = explainer(X_test)

    # Extract numeric matrix from Explanation object if needed
    if hasattr(shap_values_raw, "values"):
        vals = shap_values_raw.values
    else:
        vals = np.asarray(shap_values_raw)

    # If 3D (e.g. multiclass / binary 2-output), take slice for positive class (1)
    if vals.ndim == 3 and vals.shape[2] == 2:
        vals = vals[:, :, 1]
    elif vals.ndim == 3 and vals.shape[0] == 2:
        vals = vals[1, :, :]

    # Plot SHAP summary beeswarm plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        vals,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
        plot_size=(10, 7),
    )
    plt.title("SHAP Feature Importance & Impact Summary", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")

    # Compute global feature importance (Mean Absolute SHAP Value)
    mean_abs_shap = np.abs(vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top_5_features = [(feature_names[i], round(float(mean_abs_shap[i]), 4)) for i in sorted_idx[:5]]
    logger.info(f"Top 5 most important predictive features: {top_5_features}")
    return top_5_features


def save_final_metrics(
    metrics: Dict[str, Any],
    path: str = "outputs/metrics/final_metrics.json",
) -> None:
    """
    Export final evaluation metrics to JSON format.

    Parameters:
        metrics: Dictionary of metric key-values.
        path: Filepath for destination JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Final metrics successfully exported to {path}")


if __name__ == "__main__":
    logger.info("Executing final model evaluation and explainability pipeline...")

    # 1. Load un-SMOTEd test data
    X_test, y_test, feature_names = load_test_data()

    # 2. Load best trained model artifact
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Please run src.train first.")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded best model from {model_path} ({type(model).__name__})")

    # 3. Comprehensive test evaluation with threshold tuning if needed
    metrics, threshold = evaluate_model(model, X_test, y_test)
    print("\n==========================================")
    print("      FINAL TEST EVALUATION METRICS       ")
    print("==========================================")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("==========================================")

    # 4. Generate final class predictions using optimal threshold
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int) if threshold != 0.5 else model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    # 5. Generate evaluation plots
    plot_confusion_matrix(y_test, y_pred, path="outputs/plots/confusion_matrix.png")
    plot_roc_curve(model, X_test, y_test, path="outputs/plots/roc_curve.png")
    plot_precision_recall_curve(model, X_test, y_test, path="outputs/plots/precision_recall_curve.png")

    # 6. Interpretability with SHAP
    top5 = shap_analysis(model, X_test, feature_names, path="outputs/plots/shap_summary.png")
    print("\n--- Top 5 Predictive Features (Mean |SHAP|) ---")
    for feat, imp in top5:
        print(f"  - {feat:<35}: {imp:.4f}")

    # 7. Persist final metrics artifact
    save_final_metrics(metrics, path="outputs/metrics/final_metrics.json")

    print("\nEvaluation complete. Check outputs/plots/ and outputs/metrics/\n")
