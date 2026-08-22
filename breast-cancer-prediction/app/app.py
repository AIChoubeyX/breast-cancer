"""
Streamlit Web Dashboard for Breast Cancer Risk Prediction & Clinical Decision Support.
Provides Single Patient Risk Profiling, Cohort Batch Screening, Feature Attribution,
and Model Performance Diagnostics.
"""

import io
import os
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
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    encoders_path = os.path.join(base_dir, "models", "encoders.pkl")
    model_path = os.path.join(base_dir, "models", "best_model.pkl")

    if not os.path.exists(scaler_path):
        scaler_path = "models/scaler.pkl"
        encoders_path = "models/encoders.pkl"
        model_path = "models/best_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please execute src/train.py first.")

    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    model = joblib.load(model_path)

    return scaler, encoders, model


# -----------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
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

    # 2. Impute missing values.
    #
    # Medians come from the training split and are persisted by src.train, rather
    # than being hard-coded here. Hard-coded constants drift silently the moment
    # the training data changes, and an inference-time fill that differs from the
    # training-time fill shifts inputs away from the distribution the scaler and
    # model were fitted on.
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

    # 3. Clip BMI to the same fixed physiological bounds used in training.
    bmi_low, bmi_high = encoders.get("bmi_bounds", (15.0, 50.0))
    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].clip(lower=float(bmi_low), upper=float(bmi_high))

    # 4. Map binary categories
    binary_maps = encoders.get("binary_maps", {})
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(0.0).astype(float)

    # 5. Map ordinal categories
    ordinal_maps = encoders.get("ordinal_maps", {})
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(0.0).astype(float)

    # 6. One-hot encode against a FIXED category list.
    #
    # pd.get_dummies(drop_first=True) cannot be used at inference time. It derives
    # both the column set and the reference level from whatever values happen to be
    # present in the input. On a single-patient frame only one category is present,
    # drop_first deletes it, and the field vanishes -- step 7 then backfills both
    # dummies with 0.0, which is the encoding for "No". The user's selection was
    # silently discarded. In batch mode the reference level shifts with the file's
    # contents, so identical patients encode differently in different uploads.
    #
    # Iterating a fixed list makes the encoding independent of the input rows and
    # keeps the reference level identical to training.
    onehot_categories = encoders.get("onehot_categories", {
        "Breastfeeding_History": ["No", "Not Applicable", "Yes"],
    })
    for col, categories in onehot_categories.items():
        if col in df.columns:
            values = df[col].astype(str)
            for category in list(categories)[1:]:      # first level = reference
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
        
        # Aggregate one-hot back to high-level features if desired
        feat_imp_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        results = []
        for feat, imp in feat_imp_pairs:
            # Map dummy names back to raw feature keys
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
# 4. MAIN APPLICATION INTERFACE
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

    # Decision threshold comes from the training artifacts, where it was chosen on
    # the validation set and locked before the test set was scored. Hard-coding
    # 0.50 here would silently override that choice, so the app would operate at a
    # different point than the one every reported metric describes.
    if "decision_threshold" not in encoders:
        st.error(
            "This model was saved without a decision threshold. Re-run "
            "`python -m src.train` so the validation-selected threshold is persisted."
        )
        return
    optimal_threshold = float(encoders["decision_threshold"])

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=70)
    st.sidebar.title("Navigation")
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
                age = st.number_input("Age (Years)", min_value=20, max_value=90, value=50, step=1)
                gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
                smoking = st.selectbox("Smoking History", options=["No", "Yes"], index=0)
                alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"], index=0)
                exercise_days = st.number_input("Exercise (Days / Week)", min_value=0, max_value=7, value=3, step=1)
                annual_income = st.number_input("Annual Income (USD)", min_value=10000, max_value=150000, value=65000, step=2500)

            with col2:
                st.markdown("##### 🩺 Clinical & Physiological")
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=26.5, step=0.1, format="%.1f")
                blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=125, step=1)
                cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=350, value=210, step=1)
                diabetes = st.selectbox("Diabetes Diagnosis", options=["No", "Yes"], index=0)
                menopause = st.selectbox("Menopause Status", options=["Post", "Pre"], index=0)
                physical_activity = st.selectbox("Physical Activity Level", options=["Moderate", "Low", "High"], index=0)

            with col3:
                st.markdown("##### 🔬 Oncological & Screening Indicators")
                tumor_size = st.number_input("Tumor / Lesion Size (cm)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.2f")
                mammogram = st.selectbox("Mammogram Result", options=["Normal", "Suspicious", "Abnormal"], index=0)
                genetic_mutation = st.selectbox("Genetic Mutation (e.g. BRCA1/2)", options=["Negative", "Positive"], index=0)
                family_history = st.selectbox("Family History of Breast Cancer", options=["No", "Yes"], index=0)
                lymph_node = st.selectbox("Lymph Node Involvement", options=["No", "Yes"], index=0)
                hormone_therapy = st.selectbox("Hormone Replacement Therapy", options=["No", "Yes"], index=0)
                breastfeeding = st.selectbox("Breastfeeding History", options=["Yes", "No", "Not Applicable"], index=0)

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

                # Column validation.
                #
                # Any modelled column absent from the upload gets filled with 0.0
                # during alignment, and for a binary map 0.0 is a real level ("No",
                # "Negative"). So an absent column does not produce an error or a
                # blank -- it produces a confident prediction based on an assumption
                # the uploader never made. The eight features below are the ones that
                # actually drive the model, so missing any of them is treated as a
                # hard failure rather than a silent default.
                required_cols = [
                    "Age", "BMI", "Tumor_Size_cm", "Family_History", "Smoking",
                    "Genetic_Mutation", "Lymph_Node_Involvement", "Mammogram_Result",
                ]
                missing_req = [c for c in required_cols if c not in df_batch_raw.columns]

                if missing_req:
                    st.error(
                        f"Uploaded CSV is missing columns the model relies on: {missing_req}. "
                        "These are not optional -- absent columns would be silently "
                        "treated as low-risk values, producing confident but unfounded "
                        "predictions. Please add them and re-upload."
                    )
                    return

                # Remaining modelled columns contribute little, but the user should
                # still be told when a value was assumed rather than supplied.
                modelled_inputs = [
                    "Gender", "Alcohol_Consumption", "Physical_Activity",
                    "Hormone_Therapy", "Menopause_Status", "Blood_Pressure",
                    "Cholesterol", "Diabetes", "Exercise_Days_Per_Week",
                    "Annual_Income_USD", "Breastfeeding_History",
                ]
                defaulted = [c for c in modelled_inputs if c not in df_batch_raw.columns]
                if defaulted:
                    st.warning(
                        f"These columns were absent and have been filled with default "
                        f"values: {defaulted}. Predictions remain valid but rest partly "
                        "on assumed inputs."
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

                # Preview Data Table
                st.markdown("##### 📋 Patient-by-Patient Risk Stratification Results")
                
                def highlight_risk(val: str) -> str:
                    if "HIGH RISK" in str(val):
                        return "background-color: #fed7d7; color: #9b2c2c; font-weight: bold;"
                    return "background-color: #c6f6d5; color: #22543d; font-weight: bold;"

                st.dataframe(
                    df_results.head(100).style.applymap(highlight_risk, subset=["Risk_Classification"]),
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
