"""
Model Training and Comparison Module for Breast Cancer Prediction.
Implements Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning,
SMOTE class-imbalance mitigation, validation evaluation, and best-model selection.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.preprocess import (
    handle_missing,
    handle_outliers,
    encode_features,
    split_data,
    scale_features,
    apply_smote,
    preprocess_pipeline,
)
from src.utils import load_config, save_artifact, setup_logging

logger = setup_logging()

# Global registry of trained models in current session
_TRAINED_MODELS_CACHE: Dict[str, Any] = {}


def load_preprocessed_data(
    data_path: str = "data/breast_cancer_cleaned.csv",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load cleaned dataset and run the full preprocessing pipeline:
    missing value imputation, outlier handling, encoding, train/test split,
    feature scaling, and SMOTE oversampling on training data.

    Parameters:
        data_path: Path to cleaned dataset CSV.
        config_path: Path to config YAML.

    Returns:
        X_train, X_test, y_train, y_test, X_train_res, y_train_res
    """
    logger.info("Loading dataset and executing preprocessing pipeline...")
    return preprocess_pipeline(data_path=data_path, config_path=config_path)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model with balanced class weights.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Random state seed.

    Returns:
        Trained LogisticRegression model.
    """
    logger.info("Training Logistic Regression (class_weight='balanced', max_iter=1000)...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    _TRAINED_MODELS_CACHE["LogisticRegression"] = model
    logger.info("Logistic Regression training completed.")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with hyperparameter grid search optimizing Recall.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Random state seed.

    Returns:
        Best estimator RandomForestClassifier instance.
    """
    logger.info("Training Random Forest with GridSearchCV (optimizing Recall)...")
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="recall",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    _TRAINED_MODELS_CACHE["RandomForest"] = best_rf

    logger.info(f"Random Forest Best Params: {grid.best_params_} | Best CV Recall: {grid.best_score_:.4f}")
    return best_rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with hyperparameter grid search optimizing Recall,
    accounting for class imbalance via scale_pos_weight.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Random state seed.

    Returns:
        Best estimator XGBClassifier instance.
    """
    logger.info("Training XGBoost with GridSearchCV (optimizing Recall)...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring="recall",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    _TRAINED_MODELS_CACHE["XGBoost"] = best_xgb

    logger.info(f"XGBoost Best Params: {grid.best_params_} | Best CV Recall: {grid.best_score_:.4f}")
    return best_xgb


def evaluate_on_validation(
    models_dict: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all models on a validation set across key clinical metrics,
    prioritizing Recall for class 1 (Malignant).

    Parameters:
        models_dict: Dictionary of {model_name: trained_model_instance}.
        X_val: Validation feature matrix.
        y_val: Validation true labels.

    Returns:
        Comparison DataFrame sorted by Recall (descending).
    """
    logger.info("Evaluating models on validation dataset...")
    results = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

        # ROC-AUC score using predicted probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_val)
            auc = roc_auc_score(y_val, y_score)
        else:
            auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4),
        })

    comparison_df = pd.DataFrame(results).sort_values(by=["Recall", "ROC_AUC", "F1_Score"], ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)
    
    logger.info("Model validation comparison computed successfully.")
    return comparison_df


def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "Recall",
    models_dict: Optional[Dict[str, Any]] = None,
    save_path: str = "models/best_model.pkl",
) -> Union[str, Any]:
    """
    Select the best model based on the target metric (default: Recall for Malignant class),
    persist the model artifact to disk, and issue clinical threshold warnings if needed.

    Parameters:
        comparison_df: Comparison DataFrame from evaluate_on_validation.
        metric: Metric column name to maximize (case-insensitive).
        models_dict: Optional dictionary of model instances (defaults to internal cache).
        save_path: Destination path for best model serialization.

    Returns:
        Best model name or best model object.
    """
    # Normalize metric name to match columns
    matched_col = None
    for col in comparison_df.columns:
        if col.lower() == metric.lower():
            matched_col = col
            break
    
    if not matched_col:
        matched_col = "Recall"

    best_row = comparison_df.sort_values(by=matched_col, ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_metric_val = float(best_row[matched_col])

    logger.info(f"Selected Best Model: '{best_model_name}' based on {matched_col} = {best_metric_val:.4f}")

    # Clinical warning check
    best_recall = float(best_row.get("Recall", best_metric_val))
    if best_recall < 0.90:
        warning_msg = (
            f"WARNING: Best model '{best_model_name}' achieved Recall = {best_recall:.4f} (< 0.90 target). "
            "Clinical decision threshold tuning is recommended to reduce False Negatives!"
        )
        print(f"\n[!] {warning_msg}\n")
        logger.warning(warning_msg)

    # Save best model artifact
    models = models_dict if models_dict is not None else _TRAINED_MODELS_CACHE
    best_model_obj = models.get(best_model_name)

    if best_model_obj is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_artifact(best_model_obj, save_path)
        logger.info(f"Best model artifact saved to {save_path}")

    return best_model_name


def save_comparison_results(
    comparison_df: pd.DataFrame,
    path: str = "outputs/metrics/model_comparison.json",
) -> None:
    """
    Save model comparison metrics to JSON format.

    Parameters:
        comparison_df: Comparison DataFrame.
        path: Filepath for output JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = comparison_df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    logger.info(f"Model comparison metrics saved to {path}")


if __name__ == "__main__":
    logger.info("Starting model training pipeline...")

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_r, y_train_r = load_preprocessed_data()

    # 2. Train models on SMOTE-resampled training data
    models = {
        "LogisticRegression": train_logistic_regression(X_train_r, y_train_r),
        "RandomForest": train_random_forest(X_train_r, y_train_r),
        "XGBoost": train_xgboost(X_train_r, y_train_r),
    }

    # 3. Evaluate on original (non-SMOTE) training set as quick validation
    comparison = evaluate_on_validation(models, X_train, y_train)
    print("\n--- Model Comparison on Validation Data ---")
    print(comparison.to_string(index=False))

    # 4. Save comparison results
    save_comparison_results(comparison, path="outputs/metrics/model_comparison.json")

    # 5. Select and serialize best model
    best = select_best_model(comparison, metric="recall", models_dict=models)
    print(f"\nBest model selected and saved: {best}")
