"""
Final Test-Set Evaluation and Explainability for Breast Cancer Prediction.

This module runs EXACTLY ONE evaluation on the held-out test set, using a
decision threshold that was already locked in by src/train.py on the validation
set. Nothing here fits, tunes, selects or calibrates anything.

What that rules out, explicitly:
  - No threshold search against y_test (the previous version searched 81
    candidate thresholds on the test set and reported the winner -- that is
    test-set leakage, and the reported recall was an optimistic upper bound
    rather than an estimate of field performance).
  - No preprocessing fitted here. Imputation medians and scaler statistics are
    loaded already-fitted from the training pipeline.
  - No model refitting or model choice.

SHAP is computed post hoc purely to describe the locked model. It feeds back
into no decision, so it cannot leak. SHAP values are never reordered or filtered
by hand -- the ranking that comes out is the ranking that gets reported, however
clinically surprising it looks.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for file output

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

from src.preprocess import prepare_data
from src.utils import load_artifact, setup_logging

logger = setup_logging()

plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", palette="muted")

# Features the project brief and the review questions specifically ask about.
# Used only to annotate the reported ranking -- never to reorder it.
HIGHLIGHT_FEATURES = [
    "Age",
    "Smoking",
    "Family_History",
    "Tumor_Size_cm",
    "Genetic_Mutation",
    "BMI",
    "Mammogram_Result",
    "Lymph_Node_Involvement",
]


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------
def load_locked_threshold(encoders_path: str = "models/encoders.pkl") -> float:
    """
    Read the decision threshold that src/train.py selected on the VALIDATION set.

    Raises rather than defaulting to 0.50. A silent default would quietly
    reintroduce the bug this module exists to fix -- the operating point must be
    an artifact of training, not a constant chosen at evaluation time.
    """
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(
            f"{encoders_path} not found. Run `python -m src.train` first -- the "
            "decision threshold must come from validation, not from this script."
        )
    encoders = load_artifact(encoders_path)
    if "decision_threshold" not in encoders:
        raise KeyError(
            f"'decision_threshold' missing from {encoders_path}. Re-run src.train; "
            "evaluation refuses to invent a threshold."
        )
    threshold = float(encoders["decision_threshold"])
    logger.info(f"Loaded LOCKED decision threshold from training artifacts: {threshold}")
    return threshold


def transform_test_features(pipeline: Any, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the already-fitted imputation and scaling steps to the test features.

    Only `transform` is called -- never `fit`. The medians and the scaler mean/std
    are the ones learned from the training split, so test rows contribute nothing
    to the preprocessing. The SMOTENC step is deliberately skipped: resampling is
    a training-time device and must never touch held-out data.
    """
    X_out = pipeline.named_steps["impute"].transform(X_test)
    X_out = pipeline.named_steps["scale"].transform(X_out)
    logger.info(f"Test features transformed with train-fitted preprocessing: {X_out.shape}")
    return X_out


# -----------------------------------------------------------------------------
# Feature-name alignment (hard gate)
# -----------------------------------------------------------------------------
def get_model_feature_names(model: Any) -> Tuple[Optional[List[str]], str]:
    """
    Recover the feature names the classifier was fitted with, and say where they
    came from. Returns (names, source).
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names), "estimator.feature_names_in_"

    booster = getattr(model, "get_booster", None)
    if callable(booster):
        try:
            bnames = booster().feature_names
            if bnames:
                return list(bnames), "xgboost booster.feature_names"
        except Exception:  # pragma: no cover - booster may be unavailable
            pass

    return None, "unavailable"


def verify_feature_alignment(X_test_final: pd.DataFrame, model: Any) -> Dict[str, Any]:
    """
    Assert that the columns handed to SHAP are, in order, the columns the model was
    trained on.

    This is the check that makes the SHAP ranking trustworthy. If the order
    silently differed, every attribution would be pinned to the wrong label and
    the ranking would look arbitrary while remaining numerically valid -- a
    failure that is invisible without an explicit comparison. On mismatch this
    raises instead of warning, because a misaligned explanation is worse than no
    explanation.
    """
    data_names = list(X_test_final.columns)
    model_names, source = get_model_feature_names(model)
    n_model_features = int(getattr(model, "n_features_in_", -1))

    report: Dict[str, Any] = {
        "data_feature_count": len(data_names),
        "model_feature_count": n_model_features if n_model_features >= 0 else None,
        "model_feature_name_source": source,
        "data_feature_names": data_names,
        "model_feature_names": model_names,
    }

    if model_names is None:
        if n_model_features >= 0 and n_model_features != len(data_names):
            raise ValueError(
                f"Feature COUNT mismatch: data has {len(data_names)}, model expects "
                f"{n_model_features}. Refusing to compute SHAP."
            )
        report["aligned"] = "COUNT ONLY"
        report["exact_order_match"] = None
        logger.warning(
            "Model exposes no feature names; verified feature COUNT only "
            f"({len(data_names)} == {n_model_features})."
        )
        return report

    exact = data_names == model_names
    report["exact_order_match"] = bool(exact)
    report["aligned"] = "YES" if exact else "NO"

    if not exact:
        mismatches = [
            {"index": i, "data": d, "model": m}
            for i, (d, m) in enumerate(zip(data_names, model_names))
            if d != m
        ]
        report["mismatches"] = mismatches
        raise ValueError(
            "SHAP FEATURE ALIGNMENT FAILED -- test columns do not match the model's "
            f"training columns. First mismatches: {mismatches[:5]}. "
            "Stopping: a misaligned SHAP ranking would be silently wrong."
        )

    logger.info(
        f"SHAP alignment verified: {len(data_names)} data features == "
        f"{n_model_features} model features, exact order match (source: {source})."
    )
    return report


# -----------------------------------------------------------------------------
# Test metrics at the locked threshold
# -----------------------------------------------------------------------------
def evaluate_at_locked_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Score the model on the test set at the given threshold. One pass, no search.
    """
    logger.info(f"Single final test evaluation at locked threshold {threshold} ...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")

    metrics: Dict[str, Any] = {
        "Decision_Threshold": round(float(threshold), 3),
        "Threshold_Source": "selected on VALIDATION set in src.train (not tuned on test)",
        "Test_Set_Size": int(len(y_test)),
        "Test_Positives": int(y_test.sum()),
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
        "Average_Precision": round(float(average_precision_score(y_test, y_proba)), 4),
        "Specificity": round(specificity, 4),
        "NPV": round(npv, 4),
        "Confusion_Matrix": {
            "TN_Benign_Correct": int(tn),
            "FP_False_Alarm": int(fp),
            "FN_Missed_Cancer": int(fn),
            "TP_Malignant_Correct": int(tp),
        },
    }

    logger.info(
        f"TEST -> recall={metrics['Recall']:.4f} precision={metrics['Precision']:.4f} "
        f"F1={metrics['F1_Score']:.4f} AUC={metrics['ROC_AUC']:.4f} | "
        f"TN={tn} FP={fp} FN={fn} TP={tp}"
    )
    if metrics["Recall"] < 0.90:
        logger.warning(
            f"Test recall {metrics['Recall']:.4f} is below the 0.90 clinical target. "
            "Reported as-is; the threshold is NOT re-tuned on test data."
        )
    return metrics, y_proba, y_pred


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, threshold: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    counts = [f"{v:,}" for v in cm.flatten()]
    pcts = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]
    names = [
        "True Negative (Benign)",
        "False Positive (False Alarm)",
        "False Negative (Missed Cancer)",
        "True Positive (Malignant)",
    ]
    labels = np.asarray(
        [f"{n}\n{c}\n({p})" for n, c, p in zip(names, counts, pcts)]
    ).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
                linewidths=1.2, linecolor="black", ax=ax,
                annot_kws={"fontsize": 11})
    ax.set_title(
        f"Confusion Matrix -- Held-out Test Set (threshold = {threshold:g})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.set_xticklabels(["Benign (0)", "Malignant (1)"], fontsize=11)
    ax.set_yticklabels(["Benign (0)", "Malignant (1)"], fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def plot_roc_curve(y_test, y_proba, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#d9534f", lw=1.5, ls="--", label="Random (AUC = 0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, labelpad=10)
    ax.set_title("ROC Curve -- Held-out Test Set", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def plot_precision_recall_curve(y_test, y_proba, threshold: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    prevalence = float(np.mean(y_test))

    op_r = recall_score(y_test, (y_proba >= threshold).astype(int), zero_division=0)
    op_p = precision_score(y_test, (y_proba >= threshold).astype(int), zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#1b7837", lw=2.5, label=f"PR curve (AP = {ap:.4f})")
    ax.axhline(prevalence, color="#762a83", lw=1.5, ls="--",
               label=f"Baseline prevalence ({prevalence:.1%})")
    ax.plot([op_r], [op_p], "o", color="#d95f02", ms=11, mec="black",
            label=f"Locked threshold {threshold:g}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    ax.set_ylabel("Precision (PPV)", fontsize=12, labelpad=10)
    ax.set_title("Precision-Recall Curve -- Held-out Test Set",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------
def compute_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for the positive (malignant) class.

    Returns a 2-D array of shape (n_samples, n_features) whose column order is
    exactly X.columns -- SHAP preserves input column order, which is why
    verify_feature_alignment() is the only guarantee the labels need.
    """
    name = type(model).__name__.lower()
    if any(k in name for k in ("xgb", "forest", "tree", "boost")):
        explainer = shap.TreeExplainer(model)
    elif any(k in name for k in ("logistic", "linear", "sgd")):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)

    raw = explainer(X)
    vals = raw.values if hasattr(raw, "values") else np.asarray(raw)
    vals = np.asarray(vals)

    # Binary tree models can return one matrix per class; keep the positive class.
    if vals.ndim == 3:
        if vals.shape[2] == 2:
            vals = vals[:, :, 1]
        elif vals.shape[0] == 2:
            vals = vals[1, :, :]
        else:
            vals = vals[..., -1]

    if vals.shape != X.shape:
        raise ValueError(
            f"SHAP value matrix {vals.shape} does not match feature matrix {X.shape}."
        )
    return vals


def shap_ranking(
    vals: np.ndarray,
    feature_names: List[str],
    csv_path: str = "outputs/metrics/shap_feature_importance.csv",
) -> pd.DataFrame:
    """
    Build the COMPLETE mean|SHAP| ranking -- every feature, no truncation.

    Note on interpretation: mean|SHAP| measures average contribution across the
    cohort, so it blends effect size with prevalence. A feature that shifts risk
    sharply but only for a small subgroup ranks below a weaker feature that
    applies to everyone. That is a property of the metric, not an error, and it
    is why a low rank here is not evidence that a feature is unimportant.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    mean_abs = np.abs(vals).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs,
    })
    df = df.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    df["Pct_Of_Total"] = (df["Mean_Abs_SHAP"] / total * 100).round(2)
    df["Mean_Abs_SHAP"] = df["Mean_Abs_SHAP"].round(6)
    df["Brief_Highlighted"] = df["Feature"].isin(HIGHLIGHT_FEATURES)

    df.to_csv(csv_path, index=False)
    logger.info(f"Complete SHAP ranking ({len(df)} features) saved to {csv_path}")
    return df


def plot_shap_ranking(df: pd.DataFrame, path: str) -> None:
    """Horizontal bar chart of the complete ranking."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = df.sort_values("Mean_Abs_SHAP", ascending=True)
    colors = ["#2b5c8f" if h else "#b0b7bf" for h in d["Brief_Highlighted"]]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.36 * len(d))))
    ax.barh(d["Feature"], d["Mean_Abs_SHAP"], color=colors, edgecolor="black", lw=0.5)
    for y, v in zip(range(len(d)), d["Mean_Abs_SHAP"]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)",
                  fontsize=11, labelpad=10)
    ax.set_title("Complete SHAP Feature Importance Ranking\n"
                 "(blue = named in the project brief; order is unmodified)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.grid(True, axis="x", ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def plot_shap_beeswarm(vals: np.ndarray, X: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        vals, X, feature_names=list(X.columns),
        show=False, max_display=len(X.columns), plot_size=(10, 7),
    )
    plt.title("SHAP Summary (impact and direction)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def save_final_metrics(metrics: Dict[str, Any], path: str = "outputs/metrics/final_metrics.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Final metrics written to {path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 78)
    logger.info("FINAL TEST EVALUATION -- one pass, locked threshold, no tuning")
    logger.info("=" * 78)

    # 1. Rebuild the identical split (same random_state -> same rows as training).
    data = prepare_data()
    X_test_raw, y_test = data["X_test"], data["y_test"]

    # 2. Load training artifacts.
    pipeline_path = "models/best_pipeline.pkl"
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"{pipeline_path} not found. Run `python -m src.train` first."
        )
    pipeline = load_artifact(pipeline_path)
    model = pipeline.named_steps["clf"]
    logger.info(f"Loaded model: {type(model).__name__}")

    threshold = load_locked_threshold()

    # 3. Transform test features with train-fitted preprocessing only.
    X_test_final = transform_test_features(pipeline, X_test_raw)

    # 4. Hard alignment gate -- raises on mismatch.
    alignment = verify_feature_alignment(X_test_final, model)

    # 5. Single test evaluation.
    metrics, y_proba, y_pred = evaluate_at_locked_threshold(
        model, X_test_final, y_test, threshold
    )
    metrics["Model"] = type(model).__name__
    metrics["Feature_Alignment"] = {
        "data_feature_count": alignment["data_feature_count"],
        "model_feature_count": alignment["model_feature_count"],
        "aligned": alignment["aligned"],
        "exact_order_match": alignment["exact_order_match"],
        "source": alignment["model_feature_name_source"],
    }

    print("\n" + "=" * 58)
    print("           FINAL TEST SET METRICS")
    print("=" * 58)
    for k in ["Model", "Decision_Threshold", "Test_Set_Size", "Test_Positives",
              "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC",
              "Average_Precision", "Specificity", "NPV"]:
        print(f"  {k:<20}: {metrics[k]}")
    print("  " + "-" * 54)
    for k, v in metrics["Confusion_Matrix"].items():
        print(f"  {k:<24}: {v}")
    print("=" * 58)

    # 6. Plots.
    plot_confusion_matrix(y_test, y_pred, threshold, "outputs/plots/confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, "outputs/plots/roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, threshold,
                                "outputs/plots/precision_recall_curve.png")

    # 7. SHAP -- complete ranking, reported exactly as computed.
    vals = compute_shap_values(model, X_test_final)
    ranking = shap_ranking(vals, list(X_test_final.columns))
    plot_shap_ranking(ranking, "outputs/plots/shap_feature_importance.png")
    plot_shap_beeswarm(vals, X_test_final, "outputs/plots/shap_summary.png")

    print("\n=== COMPLETE SHAP FEATURE IMPORTANCE RANKING ===")
    print(ranking.to_string(index=False))

    print("\n--- Features named in the project brief ---")
    for feat in HIGHLIGHT_FEATURES:
        row = ranking[ranking["Feature"] == feat]
        if row.empty:
            print(f"  {feat:<26}: NOT IN MODEL (excluded from feature set)")
        else:
            r = row.iloc[0]
            print(f"  {feat:<26}: rank {int(r['Rank']):>2} of {len(ranking)}  "
                  f"mean|SHAP| = {r['Mean_Abs_SHAP']:.6f}  ({r['Pct_Of_Total']:.2f}% of total)")

    metrics["SHAP_Top_5"] = [
        {"Rank": int(r.Rank), "Feature": r.Feature, "Mean_Abs_SHAP": float(r.Mean_Abs_SHAP)}
        for r in ranking.head(5).itertuples()
    ]
    save_final_metrics(metrics)

    print("\nEvaluation complete. Test set was used exactly once, at the locked threshold.")
