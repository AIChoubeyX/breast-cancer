"""
Preprocessing module for Breast Cancer Prediction.

LEAKAGE-SAFETY DESIGN RULE
--------------------------
Transforms are split into two categories, and the distinction is what keeps this
pipeline defensible:

1. DETERMINISTIC transforms may run BEFORE the train/val/test split, because they
   use no information from the data itself -- only fixed lookup tables, fixed
   clinical bounds, or constant fills that were decided in advance:
       - dropping identifier / post-diagnostic columns
       - mapping "Yes"/"No" -> 1/0 (a fixed dictionary)
       - clipping BMI to the fixed physiological range [15, 50]
       - filling a missing categorical with a documented domain default

2. LEARNED transforms MUST be fitted on TRAINING DATA ONLY and then applied
   unchanged to validation and test:
       - median imputation of continuous features (median is a dataset statistic)
       - StandardScaler centring/scaling (mean and std are dataset statistics)
   These are implemented as scikit-learn estimators (`MedianImputerDF`,
   `StandardScalerDF`) so they can be dropped inside a cross-validation pipeline
   and re-fitted independently on every fold.

Both custom transformers operate on pandas DataFrames and preserve column order.
That matters for two reasons: SMOTENC identifies categorical features by integer
index, and SHAP attributions are labelled by column position. A ColumnTransformer
would silently reorder columns and break both.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, save_artifact, setup_logging

logger = setup_logging()


# -----------------------------------------------------------------------------
# Fixed, deterministic encoding tables.
#
# These are lookup dictionaries decided from domain semantics, NOT statistics
# estimated from the data. Applying them before the split leaks nothing.
# -----------------------------------------------------------------------------
BINARY_MAPS: Dict[str, Dict[str, int]] = {
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

ORDINAL_MAPS: Dict[str, Dict[str, int]] = {
    "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
    "Mammogram_Result": {"Normal": 0, "Suspicious": 1, "Abnormal": 2},
}

# Explicit category list so one-hot columns are identical for train, val, test and
# for the Streamlit app. pd.get_dummies() infers categories from whatever rows it
# happens to see, which is a hidden data dependency; this is deterministic.
ONEHOT_CATEGORIES: Dict[str, List[str]] = {
    "Breastfeeding_History": ["No", "Not Applicable", "Yes"],
}

# Constant fills for categorical missingness. Documented domain defaults.
CATEGORICAL_FILL_DEFAULTS: Dict[str, str] = {
    "Alcohol_Consumption": "No",
    "Physical_Activity": "Moderate",
    "Hormone_Therapy": "No",
}

# Fixed clinical bounds (not learned from the data).
BMI_BOUNDS: Tuple[float, float] = (15.0, 50.0)

DEFAULT_NUMERIC_COLS: List[str] = [
    "Age",
    "BMI",
    "Tumor_Size_cm",
    "Blood_Pressure",
    "Cholesterol",
    "Exercise_Days_Per_Week",
    "Annual_Income_USD",
]


# -----------------------------------------------------------------------------
# Learned transformers (fit on training data only)
# -----------------------------------------------------------------------------
class MedianImputerDF(BaseEstimator, TransformerMixin):
    """
    Median-impute the given continuous columns, preserving DataFrame columns/order.

    The medians are LEARNED, so they are computed in `fit` and reused verbatim in
    `transform`. Inside a cross-validation pipeline this means each fold learns
    its own medians from that fold's training portion only.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MedianImputerDF":
        X = pd.DataFrame(X)
        cols = self.columns if self.columns is not None else list(X.columns)
        self.medians_: Dict[str, float] = {
            c: float(X[c].median()) for c in cols if c in X.columns
        }
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col, median_value in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_value)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


class StandardScalerDF(BaseEstimator, TransformerMixin):
    """
    StandardScaler applied to continuous columns only, preserving DataFrame
    columns/order.

    Binary / ordinal / one-hot columns are deliberately left unscaled: they are
    passed to SMOTENC as categorical features, which requires their values to stay
    on their original discrete levels.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "StandardScalerDF":
        X = pd.DataFrame(X)
        cols = self.columns if self.columns is not None else list(X.columns)
        self.columns_ = [c for c in cols if c in X.columns]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.columns_])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X[self.columns_] = self.scaler_.transform(X[self.columns_])
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


# -----------------------------------------------------------------------------
# Deterministic cleaning and encoding (safe before the split)
# -----------------------------------------------------------------------------
def load_raw_data(data_path: str) -> pd.DataFrame:
    """Read the dataset CSV, tolerating being called from a subdirectory."""
    if not os.path.exists(data_path):
        alt = os.path.join("..", data_path)
        if os.path.exists(alt):
            data_path = alt
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {data_path}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def drop_excluded_columns(
    df: pd.DataFrame,
    excluded: List[str],
) -> pd.DataFrame:
    """
    Drop identifier and excluded (post-diagnostic / not-available-at-prediction-time)
    columns. Every exclusion is documented in DECISIONS.md.
    """
    df = df.copy()
    dropped = [c for c in excluded if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
    logger.info(f"Excluded columns dropped: {dropped}")
    return df


def clean_deterministic(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply only deterministic cleaning. Safe to run before the train/test split.

    Deliberately does NOT median-impute continuous features -- that is a learned
    statistic and is handled by `MedianImputerDF` after splitting.
    """
    df = df.copy()

    # Constant categorical fills (documented domain defaults, not statistics).
    for col, fill_value in CATEGORICAL_FILL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # Fixed physiological bound on BMI. This is a clinical rule, not a percentile
    # computed from the data, so it introduces no train/test dependency.
    if "BMI" in df.columns:
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce").clip(
            lower=BMI_BOUNDS[0], upper=BMI_BOUNDS[1]
        )

    logger.info("Deterministic cleaning completed (no learned statistics used).")
    return df


def encode_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Encode binary, ordinal, and nominal categoricals using fixed lookup tables.

    Deterministic: identical input row -> identical output row, regardless of which
    other rows are present. Therefore safe before the split.
    """
    df = df.copy()

    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).astype(float)

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).astype(float)

    # One-hot with an explicit category list (reference level = first category).
    for col, categories in ONEHOT_CATEGORIES.items():
        if col in df.columns:
            for category in categories[1:]:
                df[f"{col}_{category}"] = (
                    df[col].astype(str) == category
                ).astype(float)
            df = df.drop(columns=[col])

    logger.info(f"Deterministic encoding completed. {df.shape[1]} columns.")
    return df


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------
def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Two-stage stratified split.

        FULL -> (DEVELOPMENT 80%, TEST 20%)
        DEVELOPMENT -> (TRAIN 80% of dev, VALIDATION 20% of dev)

    With the defaults this yields 64% train / 16% validation / 20% test.

    TEST is set aside here and must not be touched again until the single final
    evaluation: not for tuning, not for model selection, not for threshold choice,
    not for fitting any transformer.
    """
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_size,
        random_state=random_state,
        stratify=y_dev if stratify else None,
    )

    total = len(X)
    logger.info(
        f"Split -> train {len(X_train)} ({len(X_train)/total:.1%}), "
        f"val {len(X_val)} ({len(X_val)/total:.1%}), "
        f"test {len(X_test)} ({len(X_test)/total:.1%})"
    )
    logger.info(
        f"Positive rate -> train {y_train.mean():.4f}, "
        f"val {y_val.mean():.4f}, test {y_test.mean():.4f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# SMOTENC support
# -----------------------------------------------------------------------------
def get_categorical_indices(
    feature_names: List[str],
    numeric_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[int]:
    """
    Compute SMOTENC categorical feature indices PROGRAMMATICALLY from the actual
    feature-name list, so the indices can never drift out of sync with the columns.

    Anything that is not a declared continuous feature is treated as categorical.
    That includes the ordinal features (Physical_Activity, Mammogram_Result): with
    only three levels each, letting SMOTENC pick an existing level is preferable to
    ordinary SMOTE inventing a fractional level like 1.37 that no patient can have.

    Returns:
        Sorted list of integer column positions of the categorical features.
    """
    numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
    numeric_present = [c for c in numeric_cols if c in feature_names]

    categorical_indices = [
        i for i, name in enumerate(feature_names) if name not in numeric_present
    ]

    if verbose:
        logger.info("SMOTENC categorical index verification:")
        for i in categorical_indices:
            logger.info(f"    index {i:>2} -> {feature_names[i]}  [categorical]")
        for name in numeric_present:
            logger.info(
                f"    index {feature_names.index(name):>2} -> {name}  [continuous]"
            )
        logger.info(
            f"    {len(categorical_indices)} categorical + "
            f"{len(numeric_present)} continuous = {len(feature_names)} features"
        )

    # Fail loudly rather than silently resampling with wrong indices.
    if len(categorical_indices) + len(numeric_present) != len(feature_names):
        raise ValueError(
            "Categorical/continuous partition does not cover all features: "
            f"{len(categorical_indices)} + {len(numeric_present)} != {len(feature_names)}"
        )

    return categorical_indices


def build_preprocessor(numeric_cols: Optional[List[str]] = None) -> SkPipeline:
    """
    Build the LEARNED preprocessing stage: median imputation then scaling of the
    continuous columns. Column names and order are preserved throughout.
    """
    numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
    return SkPipeline(
        steps=[
            ("impute", MedianImputerDF(columns=numeric_cols)),
            ("scale", StandardScalerDF(columns=numeric_cols)),
        ]
    )


def build_pipeline(
    classifier: Any,
    feature_names: List[str],
    numeric_cols: Optional[List[str]] = None,
    use_smotenc: bool = True,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> ImbPipeline:
    """
    Assemble the full modelling pipeline:

        median impute -> scale -> [SMOTENC] -> classifier

    Handing this whole object to GridSearchCV is what makes cross-validation
    honest: on every fold, imputation medians, the scaler, and SMOTENC are all
    fitted on that fold's training portion only. imblearn's Pipeline applies
    samplers during `fit` but skips them during `predict`, so the held-out fold is
    never resampled.
    """
    numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
    steps: List[Tuple[str, Any]] = [
        ("impute", MedianImputerDF(columns=numeric_cols)),
        ("scale", StandardScalerDF(columns=numeric_cols)),
    ]

    if use_smotenc:
        categorical_indices = get_categorical_indices(
            feature_names, numeric_cols=numeric_cols, verbose=False
        )
        steps.append(
            (
                "smotenc",
                SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=random_state,
                    k_neighbors=k_neighbors,
                ),
            )
        )

    steps.append(("clf", classifier))
    return ImbPipeline(steps=steps)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def prepare_data(
    data_path: str = "data/breast_cancer_cleaned.csv",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Run deterministic cleaning + encoding, then split into train / val / test.

    NOTE: this returns UNSCALED, UNIMPUTED feature frames on purpose. The learned
    preprocessing is applied later, either inside a cross-validation pipeline or
    fitted explicitly on the training split -- never on the full dataset.

    Returns a dict with the splits, the feature-name list, and metadata.
    """
    config = load_config(config_path) if os.path.exists(config_path) else {}
    features_cfg = config.get("features", {})
    model_cfg = config.get("model", {})

    df = load_raw_data(data_path)

    excluded = features_cfg.get(
        "drop", ["Patient_ID", "Biopsy_Result", "Cancer_Stage"]
    )
    df = drop_excluded_columns(df, excluded)

    df = clean_deterministic(df, config)
    df = encode_features(df, config)

    target_col = features_cfg.get("target", "Cancer")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    feature_names = list(X.columns)
    numeric_cols = [
        c for c in (features_cfg.get("numerical") or DEFAULT_NUMERIC_COLS)
        if c in feature_names
    ]

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X,
        y,
        test_size=model_cfg.get("test_size", 0.20),
        val_size=model_cfg.get("val_size", 0.20),
        random_state=config.get("project", {}).get("random_state", 42),
        stratify=model_cfg.get("stratify", True),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "excluded_columns": excluded,
        "target_col": target_col,
    }


def save_inference_artifacts(
    preprocessor: SkPipeline,
    feature_names: List[str],
    numeric_cols: List[str],
    excluded_columns: List[str],
    decision_threshold: float,
    model_dir: str = "models",
) -> None:
    """
    Persist the artifacts the Streamlit app consumes.

    The app performs its own row-wise transformation, so it needs the plain
    StandardScaler plus the encoding metadata rather than the pipeline object.
    Saving the fitted scaler out of `preprocessor` guarantees the app applies the
    exact same centring/scaling the model was trained with, and saving the learned
    medians removes the previously hard-coded imputation constants from the app.
    """
    os.makedirs(model_dir, exist_ok=True)

    imputer: MedianImputerDF = preprocessor.named_steps["impute"]
    scaler_df: StandardScalerDF = preprocessor.named_steps["scale"]

    save_artifact(scaler_df.scaler_, os.path.join(model_dir, "scaler.pkl"))

    encoders = {
        "binary_maps": BINARY_MAPS,
        "ordinal_maps": ORDINAL_MAPS,
        "onehot_categories": ONEHOT_CATEGORIES,
        "categorical_fill_defaults": CATEGORICAL_FILL_DEFAULTS,
        "bmi_bounds": list(BMI_BOUNDS),
        "feature_names": feature_names,
        "numerical_cols": numeric_cols,
        # Scaler column order, so the app scales the same columns in the same order.
        "scaler_columns": list(scaler_df.columns_),
        # Learned on the training split only; replaces the app's hard-coded values.
        "median_imputations": imputer.medians_,
        "excluded_columns": excluded_columns,
        # Selected on validation data and locked before test evaluation.
        "decision_threshold": float(decision_threshold),
    }
    save_artifact(encoders, os.path.join(model_dir, "encoders.pkl"))
    logger.info(
        f"Inference artifacts saved to {model_dir}/ "
        f"(scaler.pkl, encoders.pkl; threshold={decision_threshold:.3f})"
    )


if __name__ == "__main__":
    data = prepare_data()
    print("\n--- prepare_data() summary ---")
    print("Train:", data["X_train"].shape, "positives:", int(data["y_train"].sum()))
    print("Val:  ", data["X_val"].shape, "positives:", int(data["y_val"].sum()))
    print("Test: ", data["X_test"].shape, "positives:", int(data["y_test"].sum()))
    print("\nFeature order:")
    for i, name in enumerate(data["feature_names"]):
        print(f"  {i:>2}  {name}")
    print("\nSMOTENC categorical indices:")
    get_categorical_indices(data["feature_names"], data["numeric_cols"], verbose=True)
