"""
Streamlit Web Dashboard for Breast Cancer Risk Prediction & Clinical Decision Support.
Provides Single Patient Risk Profiling, Cohort Batch Screening, Feature Attribution,
and Model Performance Diagnostics.
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern medical UI styling
st.markdown(
    """
    <style>
    @import url('https://api.fontshare.com/v2/css?f[]=open-sauce-one@400,500,600,700,800,900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Open Sauce One', sans-serif;
    }

    .main-title {
        font-family: 'Open Sauce One', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Open Sauce One', sans-serif;
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #e53e3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(229, 62, 62, 0.2);
    }
    .low-risk-card {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border: 2px solid #38a169;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(56, 161, 105, 0.2);
    }
    .validation-alert-card {
        background: #fffaf0;
        border: 1px solid #dd6b20;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        color: #9c4221;
        margin-bottom: 1rem;
    }
    .risk-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 0.85rem 1.2rem;
        border-radius: 6px;
        color: #2b6cb0;
        font-size: 0.95rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. ARTIFACT INGESTION & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading predictive models & clinical artifacts...")
def load_artifacts() -> Tuple[Any, Dict[str, Any], Any]:
    """
    Load serialized machine learning artifacts from disk.
    Cached for optimal session performance.

    Returns:
        scaler: Fitted StandardScaler instance.
        encoders: Mapping dictionaries and feature schemas.
        model: Trained best classifier (XGBoost).
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"

    scaler_path = MODELS_DIR / "scaler.pkl"
    encoders_path = MODELS_DIR / "encoders.pkl"
    model_path = MODELS_DIR / "best_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please execute src/train.py first.")

    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    model = joblib.load(model_path)

    return scaler, encoders, model


# -----------------------------------------------------------------------------
# 3. INPUT VALIDATION & CLINICAL SANITY CHECKS
# -----------------------------------------------------------------------------
def validate_patient_inputs(inputs: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate patient features against genuine physiological and clinical bounds.

    Returns:
        errors: List of blocking error messages for impossible/invalid entries.
        warnings: List of non-blocking clinical advisory messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Demographics & Lifestyle Checks
    age = inputs.get("Age")
    if age is None or age < 18 or age > 115:
        errors.append(f"Age ({age}) is out of genuine clinical screening range [18 - 115 years].")
    elif age < 20 or age > 90:
        warnings.append(f"Age ({age} yrs) is outside the primary high-confidence model training cohort [20 - 90 yrs].")

    exercise = inputs.get("Exercise_Days_Per_Week")
    if exercise is None or exercise < 0 or exercise > 7:
        errors.append(f"Exercise days ({exercise}) must be between 0 and 7 days/week.")

    income = inputs.get("Annual_Income_USD")
    if income is None or income < 0:
        errors.append("Annual income cannot be negative.")

    # 2. Physiological & Vital Checks
    bmi = inputs.get("BMI")
    if bmi is None or bmi < 10.0 or bmi > 65.0:
        errors.append(f"BMI ({bmi} kg/m²) is outside physiologically plausible range [10.0 - 65.0 kg/m²].")
    elif bmi < 15.0 or bmi > 50.0:
        warnings.append(f"BMI ({bmi:.1f} kg/m²) will be capped to standard clinical bounds [15.0, 50.0] during inference.")

    bp = inputs.get("Blood_Pressure")
    if bp is None or bp < 70 or bp > 240:
        errors.append(f"Systolic Blood Pressure ({bp} mmHg) is outside genuine physiological range [70 - 240 mmHg].")
    elif bp > 180:
        warnings.append(f"Systolic Blood Pressure ({bp} mmHg) indicates severe hypertensive crisis level.")
    elif bp < 90:
        warnings.append(f"Systolic Blood Pressure ({bp} mmHg) indicates hypotension.")

    chol = inputs.get("Cholesterol")
    if chol is None or chol < 100 or chol > 400:
        errors.append(f"Total Cholesterol ({chol} mg/dL) is outside genuine clinical range [100 - 400 mg/dL].")
    elif chol > 300:
        warnings.append(f"Total Cholesterol ({chol} mg/dL) indicates severe hypercholesterolemia.")

    # 3. Oncological & Screening Checks
    tumor_size = inputs.get("Tumor_Size_cm")
    if tumor_size is None or tumor_size < 0.0 or tumor_size > 15.0:
        errors.append(f"Tumor Size ({tumor_size} cm) must be between 0.00 and 15.00 cm.")

    # 4. Cross-Feature Consistency Checks
    gender = inputs.get("Gender")
    breastfeeding = inputs.get("Breastfeeding_History")
    if gender == "Male" and breastfeeding == "Yes":
        warnings.append("Patient gender is 'Male' but Breastfeeding History is selected as 'Yes' (standard clinical entry is 'Not Applicable').")

    menopause = inputs.get("Menopause_Status")
    if age is not None and menopause == "Post" and age < 35:
        warnings.append(f"Patient age ({age}) is under 35 with Post-menopausal status (premature or surgical menopause).")
    elif age is not None and menopause == "Pre" and age > 65:
        warnings.append(f"Patient age ({age}) is over 65 with Pre-menopausal status.")

    lymph = inputs.get("Lymph_Node_Involvement")
    if tumor_size == 0.0 and lymph == "Yes":
        warnings.append("Lymph node involvement marked as 'Yes' with 0.0 cm primary tumor size (isolated nodal or occult primary presentation).")

    return errors, warnings


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Perform deep validation and data quality audit on an uploaded batch CSV.

    Returns:
        blocking_errors: Errors that prevent batch inference.
        data_warnings: Non-blocking data quality notices.
        audit_stats: Summary statistics of the uploaded batch.
    """
    blocking_errors: List[str] = []
    data_warnings: List[str] = []

    required_cols = [
        "Age", "BMI", "Tumor_Size_cm", "Family_History", "Smoking",
        "Genetic_Mutation", "Lymph_Node_Involvement", "Mammogram_Result",
    ]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        blocking_errors.append(
            f"Uploaded CSV is missing mandatory predictive columns: {missing_req}."
        )
        return blocking_errors, data_warnings, {}

    total_records = len(df)
    if total_records == 0:
        blocking_errors.append("Uploaded CSV contains no patient records (0 rows).")
        return blocking_errors, data_warnings, {}

    # Check for impossible negative or extreme values in numerical columns
    numeric_checks = {
        "Age": (10, 120),
        "BMI": (8.0, 75.0),
        "Blood_Pressure": (60, 260),
        "Cholesterol": (80, 500),
        "Exercise_Days_Per_Week": (0, 7),
        "Tumor_Size_cm": (0.0, 20.0),
        "Annual_Income_USD": (0, 10000000),
    }

    anomalous_counts: Dict[str, int] = {}
    for col, (vmin, vmax) in numeric_checks.items():
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = numeric_series[(numeric_series < vmin) | (numeric_series > vmax)]
            if len(out_of_bounds) > 0:
                anomalous_counts[col] = len(out_of_bounds)
                data_warnings.append(
                    f"Column '{col}' has {len(out_of_bounds)} record(s) outside expected clinical bounds [{vmin} - {vmax}]."
                )

    # Check categorical integrity
    category_valid_values = {
        "Gender": ["Female", "Male"],
        "Family_History": ["No", "Yes"],
        "Smoking": ["No", "Yes"],
        "Alcohol_Consumption": ["No", "Yes"],
        "Hormone_Therapy": ["No", "Yes"],
        "Menopause_Status": ["Pre", "Post"],
        "Genetic_Mutation": ["Negative", "Positive"],
        "Lymph_Node_Involvement": ["No", "Yes"],
        "Diabetes": ["No", "Yes"],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Mammogram_Result": ["Normal", "Suspicious", "Abnormal"],
        "Breastfeeding_History": ["No", "Yes", "Not Applicable"],
    }

    for col, valids in category_valid_values.items():
        if col in df.columns:
            non_null = df[col].dropna().astype(str).str.strip()
            invalid_mask = ~non_null.isin(valids)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                data_warnings.append(
                    f"Column '{col}' has {invalid_count} entry/entries with unrecognized categories (expected {valids})."
                )

    audit_stats = {
        "total_records": total_records,
        "features_present": len(df.columns),
        "anomalous_features": len(anomalous_counts),
    }

    return blocking_errors, data_warnings, audit_stats


# -----------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
def preprocess_input(
    df_raw: pd.DataFrame,
    scaler: Any,
    encoders: Dict[str, Any],
) -> pd.DataFrame:
    """
    Transform raw patient DataFrame to model-ready feature matrix matching training schema exactly:
    - Purges data leakage columns (Patient_ID, Biopsy_Result, Cancer_Stage)
    - Imputes missing numerical and categorical values
    - Caps extreme BMI outliers [15.0, 50.0]
    - Maps binary and ordinal categorical features
    - Encodes one-hot columns (Breastfeeding_History)
    - Re-indexes to exact training feature columns
    - Applies standard scaling on numerical columns

    Parameters:
        df_raw: Raw DataFrame from input form or batch CSV.
        scaler: Fitted StandardScaler.
        encoders: Mapping metadata.

    Returns:
        df_scaled: Cleaned, encoded, scaled feature DataFrame ready for inference.
    """
    df = df_raw.copy()

    # 1. Drop post-diagnostic data leakage columns and raw targets
    leakage_columns = ["Patient_ID", "Biopsy_Result", "Cancer_Stage", "Cancer"]
    for col in leakage_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Impute missing values
    medians = encoders.get("median_imputations", {})
    for col in ("BMI", "Tumor_Size_cm"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in medians:
                df[col] = df[col].fillna(float(medians[col]))

    categorical_fills = encoders.get("categorical_fill_defaults", {
        "Alcohol_Consumption": "No",
        "Physical_Activity": "Moderate",
        "Hormone_Therapy": "No",
    })
    for col, fill in categorical_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    # 3. Clip BMI to the same fixed physiological bounds used in training
    bmi_low, bmi_high = encoders.get("bmi_bounds", (15.0, 50.0))
    if "BMI" in df.columns:
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
        df["BMI"] = df["BMI"].clip(lower=float(bmi_low), upper=float(bmi_high))

    # 4. Map binary categories
    binary_maps = encoders.get("binary_maps", {})
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(mapping).fillna(0.0).astype(float)

    # 5. Map ordinal categories
    ordinal_maps = encoders.get("ordinal_maps", {})
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(mapping).fillna(0.0).astype(float)

    # 6. One-hot encode against a FIXED category list
    onehot_categories = encoders.get("onehot_categories", {
        "Breastfeeding_History": ["No", "Not Applicable", "Yes"],
    })
    for col, categories in onehot_categories.items():
        if col in df.columns:
            values = df[col].astype(str).str.strip()
            for category in list(categories)[1:]:  # first level = reference
                df[f"{col}_{category}"] = (values == category).astype(float)
            df = df.drop(columns=[col])

    # 7. Re-align with exact model feature names
    expected_features = encoders.get("feature_names", [])
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    df_aligned = df[expected_features].copy()

    # 8. Scale continuous numerical features
    num_cols = encoders.get("numerical_cols", [
        "Age", "BMI", "Tumor_Size_cm", "Blood_Pressure",
        "Cholesterol", "Exercise_Days_Per_Week", "Annual_Income_USD"
    ])
    cols_to_scale = [c for c in num_cols if c in df_aligned.columns]
    for c in cols_to_scale:
        df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce").fillna(medians.get(c, 0.0))
    df_aligned[cols_to_scale] = scaler.transform(df_aligned[cols_to_scale])

    return df_aligned


def get_top_contributing_features(
    model: Any,
    raw_input_dict: Dict[str, Any],
    encoders: Dict[str, Any],
    top_k: int = 3,
) -> List[Tuple[str, Any, float]]:
    """
    Extract top contributing features for individual prediction based on global feature importance.

    Parameters:
        model: Trained classifier.
        raw_input_dict: Single patient raw inputs.
        encoders: Feature metadata.
        top_k: Number of features to return.

    Returns:
        List of (feature_name, patient_value, importance_weight).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = encoders.get("feature_names", [])

        feat_imp_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        results = []
        for feat, imp in feat_imp_pairs:
            raw_key = feat.split("_Not Applicable")[0].split("_Yes")[0]
            val = raw_input_dict.get(raw_key, raw_input_dict.get(feat, "N/A"))
            results.append((raw_key, val, float(imp)))
            if len(results) >= top_k:
                break
        return results

    return [
        ("Mammogram_Result", raw_input_dict.get("Mammogram_Result", "N/A"), 0.25),
        ("Family_History", raw_input_dict.get("Family_History", "N/A"), 0.20),
        ("Genetic_Mutation", raw_input_dict.get("Genetic_Mutation", "N/A"), 0.18),
    ]


# -----------------------------------------------------------------------------
# 5. MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main() -> None:
    # Header Section
    st.markdown('<div class="main-title">🎗️ Breast Cancer Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Clinical Diagnostic Decision Support System & Early Malignancy Risk Stratification</div>',
        unsafe_allow_html=True,
    )

    try:
        scaler, encoders, model = load_artifacts()
    except Exception as e:
        st.error(f"Error loading system artifacts: {str(e)}")
        st.info("Please make sure you have run the training pipeline first: `python -m src.train`")
        return

    if "decision_threshold" not in encoders:
        st.error(
            "This model was saved without a decision threshold. Re-run "
            "`python -m src.train` so the validation-selected threshold is persisted."
        )
        return
    optimal_threshold = float(encoders["decision_threshold"])

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=70)
    st.sidebar.title("Live Demo")
    app_mode = st.sidebar.radio(
        "Select Prediction Mode:",
        ["Single Patient Prediction", "Batch Upload (CSV)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.markdown(f"**Architecture:** `{type(model).__name__}`")
    st.sidebar.markdown(f"**Decision Threshold:** `{optimal_threshold:.2f}`")
    st.sidebar.markdown("---")

    # =========================================================================
    # MODE 1: SINGLE PATIENT PREDICTION
    # =========================================================================
    if app_mode == "Single Patient Prediction":
        st.subheader("📋 Patient Diagnostic Intake Form")
        st.markdown("Enter patient demographic, lifestyle, physiological, and clinical imaging indicators below.")

        with st.form("single_patient_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographics & Lifestyle")
                age = st.number_input(
                    "Age (Years)",
                    min_value=18,
                    max_value=115,
                    value=50,
                    step=1,
                    help="Patient biological age in years (Valid range: 18 - 115).",
                )
                gender = st.selectbox(
                    "Gender",
                    options=["Female", "Male"],
                    index=0,
                    help="Biological sex of the patient.",
                )
                smoking = st.selectbox(
                    "Smoking History",
                    options=["No", "Yes"],
                    index=0,
                    help="Past or active tobacco smoking history.",
                )
                alcohol = st.selectbox(
                    "Alcohol Consumption",
                    options=["No", "Yes"],
                    index=0,
                    help="Regular alcoholic beverage consumption.",
                )
                exercise_days = st.number_input(
                    "Exercise (Days / Week)",
                    min_value=0,
                    max_value=7,
                    value=3,
                    step=1,
                    help="Number of days per week engaged in moderate-to-vigorous physical activity (0 - 7 days).",
                )
                annual_income = st.number_input(
                    "Annual Income (USD)",
                    min_value=0,
                    max_value=1000000,
                    value=65000,
                    step=2500,
                    help="Gross household annual income in USD (non-negative).",
                )

            with col2:
                st.markdown("##### 🩺 Clinical & Physiological")
                bmi = st.number_input(
                    "BMI (kg/m²)",
                    min_value=10.0,
                    max_value=60.0,
                    value=26.5,
                    step=0.1,
                    format="%.1f",
                    help="Body Mass Index in kg/m² (Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30. Outliers are bounded to [15, 50] per model protocol).",
                )
                blood_pressure = st.number_input(
                    "Systolic Blood Pressure (mmHg)",
                    min_value=70,
                    max_value=240,
                    value=125,
                    step=1,
                    help="Resting systolic blood pressure in mmHg (Normal: 90-120, Elevated: 120-129, HTN: ≥130).",
                )
                cholesterol = st.number_input(
                    "Total Cholesterol (mg/dL)",
                    min_value=100,
                    max_value=400,
                    value=210,
                    step=1,
                    help="Total serum cholesterol in mg/dL (Desirable: <200, Borderline: 200-239, High: ≥240).",
                )
                diabetes = st.selectbox(
                    "Diabetes Diagnosis",
                    options=["No", "Yes"],
                    index=0,
                    help="Clinical diagnosis of Type 1 or Type 2 Diabetes Mellitus.",
                )
                menopause = st.selectbox(
                    "Menopause Status",
                    options=["Post", "Pre"],
                    index=0,
                    help="Menopausal state (Pre-menopausal or Post-menopausal).",
                )
                physical_activity = st.selectbox(
                    "Physical Activity Level",
                    options=["Moderate", "Low", "High"],
                    index=0,
                    help="Subjective overall physical activity tier.",
                )

            with col3:
                st.markdown("##### 🔬 Oncological & Screening Indicators")
                tumor_size = st.number_input(
                    "Tumor / Lesion Size (cm)",
                    min_value=0.0,
                    max_value=15.0,
                    value=2.0,
                    step=0.1,
                    format="%.2f",
                    help="Maximum diameter of detected breast lesion/mass in cm (0.00 if no palpable/screen-detected mass).",
                )
                mammogram = st.selectbox(
                    "Mammogram Result",
                    options=["Normal", "Suspicious", "Abnormal"],
                    index=0,
                    help="Radiological mammographic screening classification.",
                )
                genetic_mutation = st.selectbox(
                    "Genetic Mutation (e.g. BRCA1/2)",
                    options=["Negative", "Positive"],
                    index=0,
                    help="High-penetrance genetic mutation screening result (e.g. BRCA1/BRCA2, PALB2, TP53).",
                )
                family_history = st.selectbox(
                    "Family History of Breast Cancer",
                    options=["No", "Yes"],
                    index=0,
                    help="First-degree relatives (mother, sister, daughter) diagnosed with breast cancer.",
                )
                lymph_node = st.selectbox(
                    "Lymph Node Involvement",
                    options=["No", "Yes"],
                    index=0,
                    help="Regional axillary or sentinel lymph node enlargement/involvement.",
                )
                hormone_therapy = st.selectbox(
                    "Hormone Replacement Therapy",
                    options=["No", "Yes"],
                    index=0,
                    help="Current or prior history of systemic hormone replacement therapy (HRT).",
                )
                breastfeeding = st.selectbox(
                    "Breastfeeding History",
                    options=["Yes", "No", "Not Applicable"],
                    index=0,
                    help="History of lactation / breastfeeding ('Not Applicable' for male patients).",
                )

            predict_submitted = st.form_submit_button("⚡ Run Risk Assessment", use_container_width=True)

        if predict_submitted:
            raw_input_dict = {
                "Age": age,
                "Gender": gender,
                "BMI": bmi,
                "Family_History": family_history,
                "Smoking": smoking,
                "Alcohol_Consumption": alcohol,
                "Physical_Activity": physical_activity,
                "Hormone_Therapy": hormone_therapy,
                "Menopause_Status": menopause,
                "Genetic_Mutation": genetic_mutation,
                "Tumor_Size_cm": tumor_size,
                "Lymph_Node_Involvement": lymph_node,
                "Mammogram_Result": mammogram,
                "Diabetes": diabetes,
                "Exercise_Days_Per_Week": exercise_days,
                "Breastfeeding_History": breastfeeding,
                "Annual_Income_USD": annual_income,
                "Blood_Pressure": blood_pressure,
                "Cholesterol": cholesterol,
            }

            # Run input validation & clinical sanity checks
            validation_errors, validation_warnings = validate_patient_inputs(raw_input_dict)

            if validation_errors:
                for err in validation_errors:
                    st.error(f"❌ Input Validation Error: {err}")
                st.stop()

            if validation_warnings:
                with st.expander("⚠️ Clinical Consistency & Boundary Notices", expanded=True):
                    for warn in validation_warnings:
                        st.warning(f"ℹ️ {warn}")

            try:
                df_single_raw = pd.DataFrame([raw_input_dict])
                df_single_proc = preprocess_input(df_single_raw, scaler, encoders)

                # Generate model probability predictions
                probabilities = model.predict_proba(df_single_proc)[0]
                prob_benign = float(probabilities[0])
                prob_malignant = float(probabilities[1])

                is_high_risk = prob_malignant >= optimal_threshold

                st.markdown("---")
                st.subheader("🎯 Risk Assessment & Diagnostic Profiling")

                # Diagnostic Status Banner
                if is_high_risk:
                    st.markdown(
                        f"""
                        <div class="high-risk-card">
                            <div class="risk-header" style="color: #c53030;">⚠️ HIGH RISK (Malignant Suspicion)</div>
                            <p style="font-size: 1.1rem; margin: 0; color: #742a2a;">
                                Patient exhibits an estimated <b>{prob_malignant * 100:.1f}%</b> probability of breast malignancy. 
                                <b>Immediate clinical follow-up and secondary diagnostic evaluation (e.g. tissue biopsy, ultrasound) are strongly recommended.</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="low-risk-card">
                            <div class="risk-header" style="color: #276749;">✅ Low Risk (Benign Profile)</div>
                            <p style="font-size: 1.1rem; margin: 0; color: #22543d;">
                                Patient exhibits an estimated <b>{prob_benign * 100:.1f}%</b> probability of benign status (Malignancy Risk: <b>{prob_malignant * 100:.1f}%</b>). 
                                Routine periodic screening schedule is recommended.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Risk Probability Progress Gauge
                st.markdown(f"**Estimated Malignancy Risk: `{prob_malignant * 100:.2f}%`** (Decision Threshold: `{optimal_threshold * 100:.1f}%`)")
                st.progress(min(max(prob_malignant, 0.0), 1.0))

                # Metric Columns
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric(label="Malignancy Probability", value=f"{prob_malignant * 100:.2f}%")
                with mcol2:
                    st.metric(label="Benign Probability", value=f"{prob_benign * 100:.2f}%")
                with mcol3:
                    st.metric(label="Clinical Triage Status", value="HIGH RISK" if is_high_risk else "LOW RISK")

                # Top 3 Contributing Features
                st.markdown("##### 🔍 Top 3 Primary Contributing Risk Drivers")
                top_features = get_top_contributing_features(model, raw_input_dict, encoders, top_k=3)

                fcol1, fcol2, fcol3 = st.columns(3)
                cols = [fcol1, fcol2, fcol3]
                for i, (feat, val, imp) in enumerate(top_features):
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="font-size: 0.85rem; color: #718096; text-transform: uppercase;">Rank #{i+1} Factor</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #2d3748; margin: 0.3rem 0;">{feat.replace('_', ' ')}</div>
                                <div style="font-size: 0.95rem; color: #4a5568;">Value: <b>{val}</b></div>
                                <div style="font-size: 0.8rem; color: #a0aec0; margin-top: 0.2rem;">Attribution Weight: {imp:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Inference error during single patient assessment: {str(e)}")

    # =========================================================================
    # MODE 2: BATCH COHORT UPLOAD
    # =========================================================================
    elif app_mode == "Batch Upload (CSV)":
        st.subheader("📁 Batch Patient Cohort Screening")
        st.markdown(
            "Upload a structured patient cohort CSV for high-throughput batch risk stratification. "
            "Post-diagnostic leakage columns (`Patient_ID`, `Biopsy_Result`, `Cancer_Stage`) will be safely stripped automatically."
        )

        uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_batch_raw = pd.read_csv(uploaded_file)
                st.success(f"File successfully loaded: `{uploaded_file.name}` ({df_batch_raw.shape[0]:,} records)")

                # Run Batch Data Quality & Range Validation Audit
                blocking_errs, audit_warns, stats = validate_batch_dataframe(df_batch_raw)

                if blocking_errs:
                    for b_err in blocking_errs:
                        st.error(f"❌ Batch Data Validation Error: {b_err}")
                    return

                if audit_warns:
                    with st.expander("⚠️ Data Quality & Clinical Range Audit", expanded=False):
                        for a_warn in audit_warns:
                            st.warning(f"ℹ️ {a_warn}")

                # Check for optional defaulted columns
                modelled_inputs = [
                    "Gender", "Alcohol_Consumption", "Physical_Activity",
                    "Hormone_Therapy", "Menopause_Status", "Blood_Pressure",
                    "Cholesterol", "Diabetes", "Exercise_Days_Per_Week",
                    "Annual_Income_USD", "Breastfeeding_History",
                ]
                defaulted = [c for c in modelled_inputs if c not in df_batch_raw.columns]
                if defaulted:
                    st.info(
                        f"Optional clinical columns absent from upload: `{defaulted}`. "
                        "Default clinical reference values will be applied."
                    )

                with st.spinner("Processing batch cohort and running inference..."):
                    df_batch_proc = preprocess_input(df_batch_raw, scaler, encoders)
                    probabilities = model.predict_proba(df_batch_proc)

                    prob_malignant = probabilities[:, 1]
                    predictions = (prob_malignant >= optimal_threshold).astype(int)

                    # Append annotated results
                    df_results = df_batch_raw.copy()
                    df_results["Malignancy_Probability"] = (prob_malignant * 100).round(2)
                    df_results["Risk_Classification"] = np.where(predictions == 1, "HIGH RISK (Malignant)", "Low Risk (Benign)")

                # Cohort Summary Statistics
                total_patients = len(df_results)
                high_risk_count = int((predictions == 1).sum())
                low_risk_count = int((predictions == 0).sum())
                high_risk_pct = (high_risk_count / total_patients) * 100
                low_risk_pct = (low_risk_count / total_patients) * 100

                st.markdown("---")
                st.markdown("##### 📊 Cohort Screening Summary")

                scol1, scol2, scol3 = st.columns(3)
                with scol1:
                    st.metric(label="Total Cohort Screened", value=f"{total_patients:,}")
                with scol2:
                    st.metric(label="High Risk Flagged", value=f"{high_risk_count:,} ({high_risk_pct:.1f}%)")
                with scol3:
                    st.metric(label="Low Risk (Benign)", value=f"{low_risk_count:,} ({low_risk_pct:.1f}%)")

                # Preview Data Table with Forward/Backward Compatible Styling
                st.markdown("##### 📋 Patient-by-Patient Risk Stratification Results")

                def highlight_risk(val: str) -> str:
                    if "HIGH RISK" in str(val):
                        return "background-color: #fed7d7; color: #9b2c2c; font-weight: bold;"
                    return "background-color: #c6f6d5; color: #22543d; font-weight: bold;"

                preview_df = df_results.head(100)
                styler = preview_df.style
                if hasattr(styler, "map"):
                    styled_preview = styler.map(highlight_risk, subset=["Risk_Classification"])
                elif hasattr(styler, "applymap"):
                    styled_preview = styler.applymap(highlight_risk, subset=["Risk_Classification"])
                else:
                    styled_preview = preview_df

                st.dataframe(
                    styled_preview,
                    use_container_width=True,
                    height=380,
                )

                # Export & Download CSV
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                st.download_button(
                    label="📥 Download Annotated Batch Predictions (CSV)",
                    data=csv_bytes,
                    file_name="breast_cancer_cohort_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error during batch CSV processing: {str(e)}")

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>⚕️ Clinical Disclaimer:</b> This tool is for decision support only. Not a medical diagnosis.
            All algorithmic risk scores must be corroborated by certified clinical practitioners and formal oncology imaging.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
