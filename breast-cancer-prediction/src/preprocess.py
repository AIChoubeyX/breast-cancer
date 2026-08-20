"""
Preprocessing module for Breast Cancer Prediction.
Handles missing value imputation, outlier capping, categorical encoding,
feature scaling, train-test splitting, and SMOTE oversampling.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, save_artifact, setup_logging

logger = setup_logging()


def handle_missing(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame according to domain rules / configuration.

    Parameters:
        df: Input DataFrame.
        config: Optional configuration dictionary.

    Returns:
        DataFrame with imputed missing values.
    """
    df = df.copy()
    
    # Impute continuous features with median
    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    if "Tumor_Size_cm" in df.columns:
        df["Tumor_Size_cm"] = df["Tumor_Size_cm"].fillna(df["Tumor_Size_cm"].median())

    # Impute categorical features with domain modes
    if "Alcohol_Consumption" in df.columns:
        df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("No")
    if "Physical_Activity" in df.columns:
        df["Physical_Activity"] = df["Physical_Activity"].fillna("Moderate")
    if "Hormone_Therapy" in df.columns:
        df["Hormone_Therapy"] = df["Hormone_Therapy"].fillna("No")

    logger.info("Missing value imputation completed.")
    return df


def handle_outliers(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Cap extreme outliers in continuous features such as BMI.

    Parameters:
        df: Input DataFrame.
        config: Optional configuration dictionary.

    Returns:
        DataFrame with capped outlier values.
    """
    df = df.copy()
    if "BMI" in df.columns:
        # Cap BMI within physiologically sensible range [15.0, 50.0]
        df["BMI"] = df["BMI"].clip(lower=15.0, upper=50.0)
    
    logger.info("Outlier handling completed.")
    return df


def encode_features(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Encode binary, ordinal, and nominal categorical features.

    Parameters:
        df: Input DataFrame.
        config: Optional configuration dictionary.

    Returns:
        DataFrame with numeric encoded features.
    """
    df = df.copy()

    # Binary feature mappings
    binary_maps = {
        "Gender": {"Female": 0, "Male": 1},
        "Family_History": {"No": 0, "Yes": 1},
        "Smoking": {"No": 0, "Yes": 1},
        "Alcohol_Consumption": {"No": 0, "Yes": 1},
        "Hormone_Therapy": {"No": 0, "Yes": 1},
        "Menopause_Status": {"Pre": 0, "Post": 1},
        "Genetic_Mutation": {"Negative": 0, "Positive": 1},
        "Lymph_Node_Involvement": {"No": 0, "Yes": 1},
        "Diabetes": {"No": 0, "Yes": 1},
    }

    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    # Ordinal feature mappings
    ordinal_maps = {
        "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
        "Mammogram_Result": {"Normal": 0, "Suspicious": 1, "Abnormal": 2},
    }

    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    # Nominal features: One-Hot Encoding
    if "Breastfeeding_History" in df.columns:
        df = pd.get_dummies(df, columns=["Breastfeeding_History"], drop_first=True, dtype=float)

    logger.info(f"Categorical encoding completed. Encoded features: {df.columns.tolist()}")
    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with stratification.

    Parameters:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of test data.
        random_state: Random state seed.
        stratify: Whether to stratify split by target.

    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    logger.info(f"Data split into Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples.")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    save_scaler_path: Optional[str] = "models/scaler.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training numerical columns and transform both train and test.

    Parameters:
        X_train: Training feature DataFrame.
        X_test: Testing feature DataFrame.
        numerical_cols: List of continuous features to scale.
        save_scaler_path: Filepath to persist fitted scaler.

    Returns:
        Scaled X_train, Scaled X_test, fitted StandardScaler instance.
    """
    if numerical_cols is None:
        numerical_cols = [
            "Age",
            "BMI",
            "Tumor_Size_cm",
            "Blood_Pressure",
            "Cholesterol",
            "Exercise_Days_Per_Week",
            "Annual_Income_USD",
        ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Scale numerical columns
    cols_to_scale = [c for c in numerical_cols if c in X_train.columns]
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        save_artifact(scaler, save_scaler_path)
        logger.info(f"Scaler saved to {save_scaler_path}")

    logger.info(f"Features scaled: {cols_to_scale}")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to oversample minority class on training partition.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Random state seed.
        k_neighbors: Number of nearest neighbours for SMOTE.

    Returns:
        Resampled feature matrix X_res, resampled target labels y_res.
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Preserve DataFrame structure with column names
    X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
    y_res_series = pd.Series(y_res, name=y_train.name)
    
    logger.info(
        f"SMOTE applied: Original counts={dict(y_train.value_counts())}, "
        f"Resampled counts={dict(y_res_series.value_counts())}"
    )
    return X_res_df, y_res_series


def preprocess_pipeline(
    data_path: str = "data/breast_cancer_cleaned.csv",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Execute full end-to-end preprocessing pipeline from raw/cleaned CSV to SMOTE-ready datasets.

    Parameters:
        data_path: Path to dataset CSV.
        config_path: Path to YAML configuration.

    Returns:
        X_train, X_test, y_train, y_test, X_train_res, y_train_res
    """
    logger.info(f"Executing preprocessing pipeline on {data_path}...")
    config = load_config(config_path) if os.path.exists(config_path) else {}

    if not os.path.exists(data_path):
        # Fallback to parent path if called from different subfolder
        if os.path.exists(os.path.join("..", data_path)):
            data_path = os.path.join("..", data_path)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Ensure leakage columns are dropped
    leakage_cols = config.get("features", {}).get("drop", ["Patient_ID", "Biopsy_Result", "Cancer_Stage"])
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Impute missing values
    df = handle_missing(df, config)

    # 3. Handle outliers
    df = handle_outliers(df, config)

    # 4. Encode categorical features
    df = encode_features(df, config)

    # 5. Separate features & target
    target_col = config.get("features", {}).get("target", "Cancer")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # 6. Train/Test Split
    test_size = config.get("model", {}).get("test_size", 0.2)
    random_state = config.get("project", {}).get("random_state", 42)
    stratify = config.get("model", {}).get("stratify", True)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # 7. Scale numerical features
    num_cols = config.get("features", {}).get("numerical", None)
    scaler_path = os.path.join(config.get("paths", {}).get("model_dir", "models/"), "scaler.pkl")
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train, X_test, numerical_cols=num_cols, save_scaler_path=scaler_path
    )

    # Save encoders mapping artifact
    encoders_dict = {
        "binary_maps": {
            "Gender": {"Female": 0, "Male": 1},
            "Family_History": {"No": 0, "Yes": 1},
            "Smoking": {"No": 0, "Yes": 1},
            "Alcohol_Consumption": {"No": 0, "Yes": 1},
            "Hormone_Therapy": {"No": 0, "Yes": 1},
            "Menopause_Status": {"Pre": 0, "Post": 1},
            "Genetic_Mutation": {"Negative": 0, "Positive": 1},
            "Lymph_Node_Involvement": {"No": 0, "Yes": 1},
            "Diabetes": {"No": 0, "Yes": 1},
        },
        "ordinal_maps": {
            "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
            "Mammogram_Result": {"Normal": 0, "Suspicious": 1, "Abnormal": 2},
        },
        "onehot_cols": ["Breastfeeding_History"],
        "feature_names": X.columns.tolist(),
        "numerical_cols": num_cols or [
            "Age", "BMI", "Tumor_Size_cm", "Blood_Pressure",
            "Cholesterol", "Exercise_Days_Per_Week", "Annual_Income_USD"
        ],
    }
    encoders_path = os.path.join(config.get("paths", {}).get("model_dir", "models/"), "encoders.pkl")
    save_artifact(encoders_dict, encoders_path)
    logger.info(f"Encoders saved to {encoders_path}")

    # 8. Apply SMOTE on training set
    smote_config = config.get("model", {}).get("smote", {})
    k_neighbors = smote_config.get("k_neighbors", 5)
    X_train_res, y_train_res = apply_smote(
        X_train_scaled, y_train, random_state=random_state, k_neighbors=k_neighbors
    )

    logger.info("Preprocessing pipeline finished successfully.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_res, y_train_res


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, X_tr_res, y_te_res = preprocess_pipeline()
    print("X_train_scaled shape:", X_tr.shape)
    print("X_test_scaled shape:", X_te.shape)
    print("X_train_resampled shape:", X_tr_res.shape)
    print("Target resampled distribution:\n", y_te_res.value_counts())
