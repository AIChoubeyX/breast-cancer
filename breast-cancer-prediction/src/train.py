"""
Model Training, Imbalance Experiments and Selection for Breast Cancer Prediction.

METHODOLOGY
-----------
Data is split 64% train / 16% validation / 20% test (stratified, random_state=42).

    TRAIN       hyperparameter search via 5-fold cross-validation
    VALIDATION  imbalance-strategy comparison, model selection, threshold choice
    TEST        untouched here -- used only by src/evaluate.py, exactly once

Two imbalance strategies are compared as CONTROLLED ALTERNATIVES rather than
stacked on top of each other (the previous version applied SMOTE *and*
class_weight='balanced' *and* scale_pos_weight simultaneously, which
double-counts the minority class and makes the effective reweighting unknowable):

    Experiment A  "class_weight"  -- cost-sensitive learning, no resampling
    Experiment B  "smotenc"       -- SMOTENC resampling, no extra class weighting

SMOTENC (not SMOTE) is used because the feature matrix is mixed-type: 13 of the
20 columns are binary / ordinal / one-hot. Plain SMOTE interpolates linearly
between neighbours, which produces values like Family_History = 0.63 or
Mammogram_Result = 1.37 -- states no patient can occupy. SMOTENC interpolates the
continuous columns and takes the majority category among neighbours for the
categorical ones.

SMOTENC sits INSIDE the cross-validation pipeline, so every fold resamples only
its own training portion. Resampling before GridSearchCV (the previous behaviour)
leaks synthetic rows -- interpolated from held-out patients -- into the validation
folds, which inflates CV scores.

Hyperparameters are tuned on average precision, not recall. Optimising recall
directly rewards a model for predicting "malignant" for everyone (recall = 1.0 at
zero clinical value). Average precision is threshold-free and focused on the
positive class, so model quality and operating point are chosen separately: the
grid search picks the model, then the decision threshold is tuned on validation
to hit the recall target.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.preprocess import (
    build_pipeline,
    build_preprocessor,
    get_categorical_indices,
    prepare_data,
    save_inference_artifacts,
)
from src.utils import save_artifact, setup_logging

logger = setup_logging()

RANDOM_STATE = 42

# Candidate decision thresholds evaluated on validation data only.
CANDIDATE_THRESHOLDS: List[float] = [0.50, 0.45, 0.40, 0.35, 0.30]

# Clinical targets. Recall on the malignant class is the primary objective; the
# precision floor stops the threshold search from collapsing to "flag everyone".
TARGET_RECALL = 0.90
MIN_PRECISION = 0.70


# -----------------------------------------------------------------------------
# Model / experiment definitions
# -----------------------------------------------------------------------------
def build_experiments(
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """
    Define the controlled imbalance experiments.

    Experiment A applies cost-sensitive weighting and no resampling.
    Experiment B applies SMOTENC and NO additional weighting, so the two
    mechanisms are never combined and their effects stay separable.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0
    logger.info(
        f"Train class balance: {neg} benign / {pos} malignant "
        f"-> scale_pos_weight = {scale_pos_weight:.4f}"
    )

    experiments: List[Dict[str, Any]] = []

    # ---- Logistic Regression -------------------------------------------------
    lr_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    experiments.append({
        "model": "LogisticRegression",
        "imbalance": "class_weight",
        "estimator": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        ),
        "param_grid": lr_grid,
        "use_smotenc": False,
    })
    experiments.append({
        "model": "LogisticRegression",
        "imbalance": "smotenc",
        "estimator": LogisticRegression(
            class_weight=None, max_iter=2000, random_state=random_state
        ),
        "param_grid": lr_grid,
        "use_smotenc": True,
    })

    # ---- Random Forest -------------------------------------------------------
    rf_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10, 15],
        "clf__min_samples_split": [2, 5],
    }
    experiments.append({
        "model": "RandomForest",
        "imbalance": "class_weight",
        "estimator": RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "param_grid": rf_grid,
        "use_smotenc": False,
    })
    experiments.append({
        "model": "RandomForest",
        "imbalance": "smotenc",
        "estimator": RandomForestClassifier(
            class_weight=None, random_state=random_state, n_jobs=-1
        ),
        "param_grid": rf_grid,
        "use_smotenc": True,
    })

    # ---- XGBoost -------------------------------------------------------------
    xgb_grid = {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1],
        "clf__n_estimators": [100, 200],
    }
    experiments.append({
        "model": "XGBoost",
        "imbalance": "class_weight",
        "estimator": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        ),
        "param_grid": xgb_grid,
        "use_smotenc": False,
    })
    experiments.append({
        "model": "XGBoost",
        # scale_pos_weight deliberately left at its default 1.0: SMOTENC has
        # already balanced the training folds, so weighting again would
        # double-count the minority class.
        "imbalance": "smotenc",
        "estimator": XGBClassifier(
            scale_pos_weight=1.0,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        ),
        "param_grid": xgb_grid,
        "use_smotenc": True,
    })

    return experiments


# -----------------------------------------------------------------------------
# Threshold selection (VALIDATION ONLY)
# -----------------------------------------------------------------------------
def select_threshold(
    y_val: pd.Series,
    y_proba_val: np.ndarray,
    candidates: Optional[List[float]] = None,
    target_recall: float = TARGET_RECALL,
    min_precision: float = MIN_PRECISION,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Choose the decision threshold using VALIDATION data only.

    Documented rule, applied deterministically:
      1. Keep candidates whose validation precision >= min_precision.
      2. Among those, take the highest recall (missed cancers are the costly error).
      3. Break ties on F1, then on the higher threshold (fewer false alarms).
      4. If no candidate clears the precision floor, fall back to max F1 and warn.

    Returns the chosen threshold and the full sweep table for reporting.
    """
    candidates = candidates or CANDIDATE_THRESHOLDS

    sweep: List[Dict[str, float]] = []
    for t in candidates:
        y_pred = (y_proba_val >= t).astype(int)
        sweep.append({
            "Threshold": round(float(t), 3),
            "Precision": round(float(precision_score(y_val, y_pred, pos_label=1, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_val, y_pred, pos_label=1, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_val, y_pred, pos_label=1, zero_division=0)), 4),
            "Accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        })

    eligible = [r for r in sweep if r["Precision"] >= min_precision]
    if eligible:
        best = sorted(
            eligible,
            key=lambda r: (r["Recall"], r["F1_Score"], r["Threshold"]),
            reverse=True,
        )[0]
    else:
        best = sorted(sweep, key=lambda r: (r["F1_Score"], r["Threshold"]), reverse=True)[0]
        logger.warning(
            f"No candidate threshold reached precision >= {min_precision}. "
            f"Falling back to max-F1 threshold {best['Threshold']}."
        )

    chosen = float(best["Threshold"])
    if best["Recall"] < target_recall:
        logger.warning(
            f"Validation recall at chosen threshold {chosen} is {best['Recall']:.4f}, "
            f"below the {target_recall} target."
        )
    logger.info(
        f"Threshold selected on VALIDATION: {chosen} "
        f"(recall={best['Recall']:.4f}, precision={best['Precision']:.4f}, F1={best['F1_Score']:.4f})"
    )
    return chosen, sweep


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------
def run_experiment(
    experiment: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    numeric_cols: List[str],
    cv_folds: int = 5,
    scoring: str = "average_precision",
) -> Dict[str, Any]:
    """
    Tune one model/imbalance combination with cross-validation on TRAIN, then score
    it on VALIDATION.

    The whole pipeline (impute -> scale -> optional SMOTENC -> classifier) goes into
    GridSearchCV, so every fold fits its own imputation medians, its own scaler and
    its own SMOTENC on that fold's training portion only.
    """
    label = f"{experiment['model']} [{experiment['imbalance']}]"
    logger.info(f"--- Running experiment: {label} ---")

    pipeline = build_pipeline(
        classifier=experiment["estimator"],
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        use_smotenc=experiment["use_smotenc"],
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=experiment["param_grid"],
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring,
        n_jobs=1,          # inner estimators already use n_jobs=-1
        refit=True,
        verbose=0,
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    best_pipeline = grid.best_estimator_

    t0 = time.perf_counter()
    y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]
    inference_seconds = time.perf_counter() - t0

    # Threshold chosen on validation, per experiment.
    threshold, sweep = select_threshold(y_val, y_proba_val)
    y_pred_val = (y_proba_val >= threshold).astype(int)

    # Also record the default operating point for a like-for-like comparison.
    y_pred_default = (y_proba_val >= 0.50).astype(int)

    result = {
        "Model": experiment["model"],
        "Imbalance_Method": experiment["imbalance"],
        "Best_Params": {k.replace("clf__", ""): v for k, v in grid.best_params_.items()},
        "CV_Metric": scoring,
        "CV_Score_Mean": round(float(grid.best_score_), 4),
        "CV_Score_Std": round(
            float(grid.cv_results_["std_test_score"][grid.best_index_]), 4
        ),
        "Threshold": round(float(threshold), 3),
        "Val_Accuracy": round(float(accuracy_score(y_val, y_pred_val)), 4),
        "Val_Precision": round(float(precision_score(y_val, y_pred_val, pos_label=1, zero_division=0)), 4),
        "Val_Recall": round(float(recall_score(y_val, y_pred_val, pos_label=1, zero_division=0)), 4),
        "Val_F1": round(float(f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)), 4),
        "Val_ROC_AUC": round(float(roc_auc_score(y_val, y_proba_val)), 4),
        "Val_Avg_Precision": round(float(average_precision_score(y_val, y_proba_val)), 4),
        "Val_Recall_at_0.50": round(float(recall_score(y_val, y_pred_default, pos_label=1, zero_division=0)), 4),
        "Val_Precision_at_0.50": round(float(precision_score(y_val, y_pred_default, pos_label=1, zero_division=0)), 4),
        "Train_Time_Seconds": round(train_seconds, 2),
        "Inference_Time_ms_per_1k": round(
            inference_seconds / max(len(X_val), 1) * 1_000_000, 3
        ),
        "Threshold_Sweep": sweep,
    }

    logger.info(
        f"{label}: CV {scoring}={result['CV_Score_Mean']:.4f} | "
        f"val recall={result['Val_Recall']:.4f} precision={result['Val_Precision']:.4f} "
        f"F1={result['Val_F1']:.4f} AUC={result['Val_ROC_AUC']:.4f} @t={result['Threshold']}"
    )

    return {"summary": result, "pipeline": best_pipeline}


def select_best_experiment(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pick the winning configuration using VALIDATION metrics only, in the priority
    order the project's clinical objective implies: malignant recall, then F1, then
    ROC-AUC (accuracy is deliberately not a criterion -- at 19.3% prevalence,
    predicting "benign" for everyone scores 80.7%).
    """
    ranked = sorted(
        results,
        key=lambda r: (
            r["summary"]["Val_Recall"],
            r["summary"]["Val_F1"],
            r["summary"]["Val_ROC_AUC"],
        ),
        reverse=True,
    )
    best = ranked[0]
    logger.info(
        f"Selected: {best['summary']['Model']} [{best['summary']['Imbalance_Method']}] "
        f"-- validation recall {best['summary']['Val_Recall']:.4f}"
    )
    return best


def comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the sorted model-comparison table for reporting."""
    rows = [r["summary"] for r in results]
    df = pd.DataFrame([
        {k: v for k, v in row.items() if k != "Threshold_Sweep"} for row in rows
    ])
    df = df.sort_values(
        by=["Val_Recall", "Val_F1", "Val_ROC_AUC"], ascending=False
    ).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 78)
    logger.info("TRAINING PIPELINE -- leakage-safe, test set untouched")
    logger.info("=" * 78)

    # 1. Deterministic cleaning + encoding, then the three-way split.
    data = prepare_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    feature_names = data["feature_names"]
    numeric_cols = data["numeric_cols"]

    logger.info(f"Features ({len(feature_names)}): {feature_names}")
    categorical_indices = get_categorical_indices(feature_names, numeric_cols, verbose=True)

    # 2. Controlled imbalance experiments.
    experiments = build_experiments(y_train)
    results: List[Dict[str, Any]] = []
    for experiment in experiments:
        results.append(
            run_experiment(
                experiment,
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names,
                numeric_cols=numeric_cols,
            )
        )

    # 3. Comparison table (validation only).
    table = comparison_table(results)
    print("\n=== MODEL COMPARISON (VALIDATION SET) ===")
    print(table.to_string(index=False))

    # 4. Select the winner and lock its threshold.
    best = select_best_experiment(results)
    best_summary = best["summary"]
    best_pipeline = best["pipeline"]
    locked_threshold = float(best_summary["Threshold"])

    print("\n=== THRESHOLD SWEEP (VALIDATION) FOR SELECTED MODEL ===")
    print(pd.DataFrame(best_summary["Threshold_Sweep"]).to_string(index=False))

    # 5. Persist artifacts.
    #
    # The Streamlit app applies imputation and scaling itself and then calls
    # predict_proba on the resulting frame. So we save the plain classifier plus
    # the fitted scaler and the learned medians, which together reproduce exactly
    # the transformation the classifier was trained under.
    os.makedirs("models", exist_ok=True)

    final_classifier = best_pipeline.named_steps["clf"]
    save_artifact(final_classifier, "models/best_model.pkl")

    # Refit a standalone preprocessor on TRAIN only, matching the pipeline's
    # first two steps, so its fitted state can be exported for inference.
    preprocessor = build_preprocessor(numeric_cols=numeric_cols)
    preprocessor.fit(X_train)

    save_inference_artifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        excluded_columns=data["excluded_columns"],
        decision_threshold=locked_threshold,
        model_dir="models",
    )

    # Save the whole fitted pipeline too -- evaluate.py uses it so that test-set
    # preprocessing is byte-identical to what the model was trained with.
    save_artifact(best_pipeline, "models/best_pipeline.pkl")

    # 6. Comparison + selection record.
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluated_on": "validation set (16% of full data); test set untouched",
                "cv_metric": "average_precision (5-fold StratifiedKFold on TRAIN only)",
                "selection_rule": "max validation recall, then F1, then ROC-AUC",
                "split_sizes": {
                    "train": int(len(X_train)),
                    "validation": int(len(X_val)),
                    "test": int(len(data["X_test"])),
                },
                "feature_names": feature_names,
                "smotenc_categorical_indices": categorical_indices,
                "selected": {
                    "model": best_summary["Model"],
                    "imbalance_method": best_summary["Imbalance_Method"],
                    "best_params": best_summary["Best_Params"],
                    "locked_threshold": locked_threshold,
                },
                "experiments": [r["summary"] for r in results],
            },
            f,
            indent=4,
        )
    logger.info("Model comparison saved to outputs/metrics/model_comparison.json")

    print(f"\nSelected model : {best_summary['Model']} [{best_summary['Imbalance_Method']}]")
    print(f"Best params    : {best_summary['Best_Params']}")
    print(f"Locked threshold (from VALIDATION): {locked_threshold}")
    print("\nTest set has NOT been touched. Run `python -m src.evaluate` for final metrics.")
