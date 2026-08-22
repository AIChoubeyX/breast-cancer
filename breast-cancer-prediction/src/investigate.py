"""
Diagnostic investigations for the breast cancer pipeline.

Answers three questions that came out of reviewing the SHAP output, plus one
question nobody asked but that turns out to govern how every other number in
this project should be read.

  1. Why does Smoking rank above Age, when clinical literature puts smoking's
     association with breast cancer as weak and still debated?
  2. Is Lymph_Node_Involvement acting as a proxy for the diagnosis, the way
     Biopsy_Result and Cancer_Stage were?
  3. Why are Tumor_Size_cm and Genetic_Mutation absent from the SHAP top 5 when
     the brief names them as key biomarkers?
  4. (Unprompted) What actually generates the target column?

Every statistic here is computed on the DEV split only (train + validation).
The test set is never read, not even for descriptive statistics -- findings here
could influence feature decisions, so letting them see test data would leak.

This module only measures. It does not modify the feature set, the model, or the
SHAP ranking.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.preprocess import prepare_data
from src.utils import setup_logging

logger = setup_logging()

RANDOM_STATE = 42

# A feature whose single-column AUC reaches this is doing the job of the label.
LEAKAGE_AUC_FLAG = 0.95

# The eight columns that carry signal, per the association screen below.
RULE_FEATURES = [
    "Age", "BMI", "Tumor_Size_cm", "Family_History",
    "Smoking", "Genetic_Mutation", "Lymph_Node_Involvement", "Mammogram_Result",
]


# -----------------------------------------------------------------------------
# Association measures
# -----------------------------------------------------------------------------
def cramers_v(x: pd.Series, y: pd.Series, bias_correction: bool = True) -> float:
    """
    Cramer's V between two categorical series, with Bergsma's bias correction.

    Without the correction, V is biased upward for sparse tables, which makes
    pure-noise columns look weakly predictive. With it, genuinely independent
    columns fall to (or clamp at) exactly zero -- useful for telling a weak real
    signal apart from no signal at all.
    """
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape

    if not bias_correction:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))

    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def univariate_auc(feature: pd.Series, y: pd.Series) -> float:
    """
    Single-feature ROC-AUC, oriented so the result is always >= 0.50.

    Comparable across continuous and categorical features, which Cramer's V and
    point-biserial r are not -- so this is the metric used for the leakage screen.
    """
    from sklearn.metrics import roc_auc_score

    if feature.dtype == object or isinstance(feature.dtype, pd.CategoricalDtype) or feature.dtype == "str":
        # Order levels by observed positive rate so an arbitrary alphabetical
        # encoding cannot understate a genuinely predictive column.
        rates = y.groupby(feature).mean()
        scores = feature.map(rates).astype(float)
    else:
        scores = feature.astype(float)

    mask = scores.notna() & y.notna()
    if mask.sum() < 10 or y[mask].nunique() < 2:
        return float("nan")
    auc = roc_auc_score(y[mask], scores[mask])
    return float(max(auc, 1 - auc))


def association_screen(df: pd.DataFrame, target: str = "Cancer") -> pd.DataFrame:
    """
    Rank every candidate feature by its association with the target, and flag
    anything strong enough to be a label proxy rather than a risk factor.
    """
    rows: List[Dict[str, Any]] = []
    y = df[target].astype(int)

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        is_cat = s.dtype == object or s.dtype == "str" or s.nunique() <= 10

        if is_cat:
            assoc = cramers_v(s.astype(str), y.astype(str))
            assoc_type = "cramers_v"
            table = pd.crosstab(s.astype(str), y)
            chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        else:
            valid = s.notna()
            assoc, p = stats.pointbiserialr(y[valid], s[valid].astype(float))
            assoc = abs(float(assoc))
            assoc_type = "point_biserial_r"
            chi2 = float("nan")

        auc = univariate_auc(s, y)
        rows.append({
            "Feature": col,
            "Association": round(float(assoc), 6),
            "Association_Type": assoc_type,
            "Univariate_AUC": round(auc, 6) if not np.isnan(auc) else None,
            "Chi2": round(float(chi2), 2) if not np.isnan(chi2) else None,
            "P_Value": float(p),
            "Leakage_Flag": bool((not np.isnan(auc)) and auc >= LEAKAGE_AUC_FLAG),
        })

    out = pd.DataFrame(rows).sort_values("Association", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


# -----------------------------------------------------------------------------
# Leakage evidence for a single categorical feature
# -----------------------------------------------------------------------------
def proxy_evidence(feature: pd.Series, y: pd.Series, name: str) -> Dict[str, Any]:
    """
    Test whether a categorical feature behaves like a restatement of the label.

    The decisive quantity is the off-diagonal count. A genuine risk factor has
    plenty of patients who carry it without the disease. A label proxy has
    (almost) none -- that is what made Biopsy_Result detectable. Correlation
    strength alone cannot separate the two cases; an empty off-diagonal can.
    """
    s = feature.astype(str)
    table = pd.crosstab(s, y)
    levels: Dict[str, Any] = {}

    for level in table.index:
        n = int(table.loc[level].sum())
        pos = int(table.loc[level].get(1, 0))
        levels[str(level)] = {
            "n": n,
            "positives": pos,
            "positive_rate": round(pos / n, 6) if n else None,
            "negatives_with_this_level": int(table.loc[level].get(0, 0)),
        }

    total_pos = int(y.sum())
    # Highest-risk level: how many disease-free patients still carry it?
    top_level = max(levels, key=lambda k: levels[k]["positive_rate"] or 0)
    counter_examples = levels[top_level]["negatives_with_this_level"]
    captured = levels[top_level]["positives"] / total_pos if total_pos else float("nan")

    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n_total = int(table.to_numpy().sum())

    return {
        "feature": name,
        "levels": levels,
        "highest_risk_level": top_level,
        "counter_examples_at_highest_risk_level": counter_examples,
        "counter_example_share_of_that_level": round(
            counter_examples / levels[top_level]["n"], 6
        ) if levels[top_level]["n"] else None,
        "share_of_all_cancers_captured": round(float(captured), 6),
        "cramers_v": round(cramers_v(s, y.astype(str)), 6),
        "univariate_auc": round(univariate_auc(feature, y), 6),
        "chi2": round(float(chi2), 2),
        "chi2_equals_n": bool(abs(float(chi2) - n_total) < 1.0),
        "n": n_total,
        "verdict": (
            "DETERMINISTIC PROXY -- restates the label"
            if counter_examples == 0
            else "RISK FACTOR -- many counter-examples exist"
        ),
    }


# -----------------------------------------------------------------------------
# Multivariate importance via leave-one-feature-out
# -----------------------------------------------------------------------------
def encode_for_model(df: pd.DataFrame, target: str = "Cancer") -> Tuple[pd.DataFrame, pd.Series]:
    """Numeric matrix for the ablation study: rate-ordered categoricals, median-filled."""
    y = df[target].astype(int)
    X = df.drop(columns=[target]).copy()
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype == "str":
            rates = y.groupby(X[col].astype(str)).mean()
            X[col] = X[col].astype(str).map(rates).astype(float)
    return X.fillna(X.median(numeric_only=True)), y


def ablation_study(
    df: pd.DataFrame,
    features: List[str],
    target: str = "Cancer",
    folds: int = 5,
) -> pd.DataFrame:
    """
    Measure each feature's marginal contribution by removing it and re-scoring.

    Univariate association answers "how much does this column know on its own".
    This answers "how much does the model lose without it" -- which is the
    question a SHAP ranking is actually reflecting, and the two can disagree when
    features carry overlapping information.
    """
    X, y = encode_for_model(df, target)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    def score(cols: List[str]) -> Tuple[float, float]:
        Xs = StandardScaler().fit_transform(X[cols])
        s = cross_val_score(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            Xs, y, cv=cv, scoring="roc_auc",
        )
        return float(s.mean()), float(s.std())

    all_cols = list(X.columns)
    base_mean, base_std = score(all_cols)
    logger.info(f"Ablation baseline (all {len(all_cols)} features): AUC {base_mean:.6f} +/- {base_std:.6f}")

    rows = [{
        "Feature": "(baseline: all features)",
        "AUC_Without_Feature": round(base_mean, 6),
        "AUC_Drop": 0.0,
        "CV_Std": round(base_std, 6),
    }]
    for f in features:
        if f not in all_cols:
            continue
        m, sd = score([c for c in all_cols if c != f])
        rows.append({
            "Feature": f,
            "AUC_Without_Feature": round(m, 6),
            "AUC_Drop": round(base_mean - m, 6),
            "CV_Std": round(sd, 6),
        })

    out = pd.DataFrame(rows)
    body = out.iloc[1:].sort_values("AUC_Drop", ascending=False)
    return pd.concat([out.iloc[:1], body]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Provenance: is the target a deterministic function of the features?
# -----------------------------------------------------------------------------
def generative_rule_check(df: pd.DataFrame, target: str = "Cancer") -> Dict[str, Any]:
    """
    Test the hypothesis that the label is a threshold count over eight binary
    conditions, rather than a noisy probabilistic outcome.

    This is the single most consequential fact about the dataset. If it holds,
    the very high AUC this project reports is not evidence of a good model and
    not evidence of leakage either -- it is an artifact of a synthetic generator
    that left almost no irreducible noise to lose. It also explains the SHAP
    ranking: the generator weights its eight conditions equally, so the ordering
    reflects how often each condition fires, not how much any of them matters
    clinically.
    """
    conditions = {
        "Age>=50": df["Age"] >= 50,
        "BMI>=30": df["BMI"] >= 30,
        "Tumor_Size_cm>=3.0": df["Tumor_Size_cm"] >= 3.0,
        "Family_History==Yes": df["Family_History"].astype(str) == "Yes",
        "Smoking==Yes": df["Smoking"].astype(str) == "Yes",
        "Genetic_Mutation==Positive": df["Genetic_Mutation"].astype(str) == "Positive",
        "Lymph_Node_Involvement==Yes": df["Lymph_Node_Involvement"].astype(str) == "Yes",
        "Mammogram_Result!=Normal": df["Mammogram_Result"].astype(str) != "Normal",
    }
    cond_df = pd.DataFrame(conditions)
    score = cond_df.fillna(False).astype(int).sum(axis=1)
    y = df[target].astype(int)

    predicted = (score >= 4).astype(int)
    complete = df[list({"Age", "BMI", "Tumor_Size_cm", "Family_History", "Smoking",
                        "Genetic_Mutation", "Lymph_Node_Involvement",
                        "Mammogram_Result"})].notna().all(axis=1)

    by_score = (
        pd.DataFrame({"score": score, "y": y})
        .groupby("score")["y"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "cancer_rate"})
    )

    return {
        "hypothesis": "Cancer = 1 iff at least 4 of 8 binary risk conditions hold",
        "conditions": list(conditions.keys()),
        "accuracy_all_rows": round(float((predicted == y).mean()), 6),
        "accuracy_complete_cases": round(float((predicted[complete] == y[complete]).mean()), 6),
        "n_complete_cases": int(complete.sum()),
        "cancer_rate_by_score": {
            int(i): {"n": int(r["n"]), "cancer_rate": round(float(r["cancer_rate"]), 6)}
            for i, r in by_score.iterrows()
        },
        "condition_prevalence": {
            k: round(float(v.fillna(False).mean()), 6) for k, v in conditions.items()
        },
    }


def prevalence_lift_decomposition(df: pd.DataFrame, target: str = "Cancer") -> pd.DataFrame:
    """
    Separate how HARD a feature pushes risk from how OFTEN it applies.

    mean|SHAP| is an average over the whole cohort, so it multiplies these two
    together. A feature can push risk very hard for a small group and still rank
    low, which is exactly the Genetic_Mutation case. Reporting the factors
    separately keeps a low SHAP rank from being misread as "not predictive".
    """
    y = df[target].astype(int)
    base = float(y.mean())
    rows = []

    for col, positive_level in [
        ("Genetic_Mutation", "Positive"),
        ("Family_History", "Yes"),
        ("Smoking", "Yes"),
        ("Lymph_Node_Involvement", "Yes"),
    ]:
        mask = df[col].astype(str) == positive_level
        rows.append({
            "Feature": f"{col}=={positive_level}",
            "Prevalence": round(float(mask.mean()), 6),
            "P_Cancer_Given_Present": round(float(y[mask].mean()), 6),
            "P_Cancer_Given_Absent": round(float(y[~mask].mean()), 6),
            "Risk_Ratio": round(float(y[mask].mean() / y[~mask].mean()), 6),
            "Cancers_Captured": int(y[mask].sum()),
            "Share_Of_All_Cancers": round(float(y[mask].sum() / y.sum()), 6),
        })

    for col, thresh in [("Age", 50), ("BMI", 30), ("Tumor_Size_cm", 3.0)]:
        mask = df[col] >= thresh
        mask = mask.fillna(False)
        rows.append({
            "Feature": f"{col}>={thresh}",
            "Prevalence": round(float(mask.mean()), 6),
            "P_Cancer_Given_Present": round(float(y[mask].mean()), 6),
            "P_Cancer_Given_Absent": round(float(y[~mask].mean()), 6),
            "Risk_Ratio": round(float(y[mask].mean() / y[~mask].mean()), 6),
            "Cancers_Captured": int(y[mask].sum()),
            "Share_Of_All_Cancers": round(float(y[mask].sum() / y.sum()), 6),
        })

    mask = df["Mammogram_Result"].astype(str) != "Normal"
    rows.append({
        "Feature": "Mammogram_Result!=Normal",
        "Prevalence": round(float(mask.mean()), 6),
        "P_Cancer_Given_Present": round(float(y[mask].mean()), 6),
        "P_Cancer_Given_Absent": round(float(y[~mask].mean()), 6),
        "Risk_Ratio": round(float(y[mask].mean() / y[~mask].mean()), 6),
        "Cancers_Captured": int(y[mask].sum()),
        "Share_Of_All_Cancers": round(float(y[mask].sum() / y.sum()), 6),
    })

    out = pd.DataFrame(rows).sort_values("Risk_Ratio", ascending=False).reset_index(drop=True)
    out.attrs["base_rate"] = round(base, 6)
    return out


def encoding_order_check(df: pd.DataFrame, target: str = "Cancer") -> Dict[str, Any]:
    """
    Check whether the ordinal encoding of Mammogram_Result respects its risk order.

    Alphabetical ordering puts Normal -- the lowest-risk level -- in the middle,
    which destroys monotonicity and understates the feature badly for any model
    that treats the code as a number.
    """
    y = df[target].astype(int)
    rates = y.groupby(df["Mammogram_Result"].astype(str)).mean().sort_values()
    alpha = sorted(rates.index)

    alpha_codes = df["Mammogram_Result"].astype(str).map({v: i for i, v in enumerate(alpha)})
    rate_codes = df["Mammogram_Result"].astype(str).map({v: i for i, v in enumerate(rates.index)})

    from sklearn.metrics import roc_auc_score
    return {
        "cancer_rate_by_level": {k: round(float(v), 6) for k, v in rates.items()},
        "alphabetical_order": alpha,
        "risk_ascending_order": list(rates.index),
        "auc_alphabetical_encoding": round(float(max(
            roc_auc_score(y, alpha_codes), 1 - roc_auc_score(y, alpha_codes))), 6),
        "auc_risk_ordered_encoding": round(float(max(
            roc_auc_score(y, rate_codes), 1 - roc_auc_score(y, rate_codes))), 6),
        "configured_order": ["Normal", "Suspicious", "Abnormal"],
        "configured_order_is_monotone_in_risk": bool(
            list(rates.index) == ["Normal", "Suspicious", "Abnormal"]
        ),
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 78)
    logger.info("DIAGNOSTIC INVESTIGATIONS (dev split only -- test set never read)")
    logger.info("=" * 78)

    # Rebuild the dev split from the same seed, then recover the ORIGINAL
    # (unencoded) rows for those indices so categorical levels stay readable.
    data = prepare_data()
    dev_index = list(data["X_train"].index) + list(data["X_val"].index)

    raw = pd.read_csv("data/breast_cancer_cleaned.csv")
    dev = raw.loc[dev_index].copy()
    logger.info(f"Dev split: {len(dev)} rows ({dev['Cancer'].sum()} positive). Test rows excluded.")

    results: Dict[str, Any] = {
        "scope": "train + validation only (8000 rows); test set excluded from all statistics",
        "n_rows": int(len(dev)),
        "n_positive": int(dev["Cancer"].sum()),
        "base_rate": round(float(dev["Cancer"].mean()), 6),
    }

    # --- Association screen over every candidate feature ---------------------
    screen = association_screen(dev)
    results["association_screen"] = screen.to_dict(orient="records")
    print("\n=== ASSOCIATION SCREEN (all candidate features, dev split) ===")
    print(screen.to_string(index=False))

    flagged = screen[screen["Leakage_Flag"]]
    results["leakage_flagged_features"] = flagged["Feature"].tolist()
    print(f"\nFeatures with univariate AUC >= {LEAKAGE_AUC_FLAG}: "
          f"{flagged['Feature'].tolist() or 'NONE'}")

    # --- Q2: is Lymph_Node_Involvement a diagnosis proxy? -------------------
    print("\n=== INVESTIGATION: Lymph_Node_Involvement as a possible label proxy ===")
    lymph = proxy_evidence(dev["Lymph_Node_Involvement"], dev["Cancer"].astype(int),
                          "Lymph_Node_Involvement")
    results["lymph_node_investigation"] = lymph
    print(json.dumps(lymph, indent=2))

    # Benchmark against a column known to be a perfect proxy.
    raw_full = pd.read_csv("data/breast_cancer_prediction.csv")
    if len(raw_full) == len(raw) and (raw_full["Age"].values == raw["Age"].values).all():
        biopsy_dev = raw_full.loc[dev_index, "Biopsy_Result"]
        biopsy = proxy_evidence(biopsy_dev, dev["Cancer"].astype(int), "Biopsy_Result (dropped)")
        results["biopsy_result_benchmark"] = biopsy
        print("\n--- Benchmark: Biopsy_Result, a known deterministic proxy ---")
        print(json.dumps(biopsy, indent=2))
    else:
        logger.warning("Raw and cleaned files are not positionally aligned; "
                       "skipping the Biopsy_Result benchmark.")
        results["biopsy_result_benchmark"] = None

    # --- Q1: Smoking vs Age -------------------------------------------------
    print("\n=== INVESTIGATION: Smoking ranking above Age ===")
    smoking_vs_age = {
        "smoking": proxy_evidence(dev["Smoking"], dev["Cancer"].astype(int), "Smoking"),
        "age_univariate_auc": round(univariate_auc(dev["Age"], dev["Cancer"].astype(int)), 6),
        "smoking_univariate_auc": round(univariate_auc(dev["Smoking"], dev["Cancer"].astype(int)), 6),
    }
    results["smoking_vs_age"] = smoking_vs_age
    print(f"Univariate AUC -- Age {smoking_vs_age['age_univariate_auc']:.6f} vs "
          f"Smoking {smoking_vs_age['smoking_univariate_auc']:.6f}")

    # --- Multivariate ablation ---------------------------------------------
    print("\n=== ABLATION: marginal contribution of each signal feature ===")
    ablation = ablation_study(dev, RULE_FEATURES)
    results["ablation_study"] = ablation.to_dict(orient="records")
    print(ablation.to_string(index=False))

    # --- Q3: prevalence vs lift -------------------------------------------
    print("\n=== PREVALENCE vs LIFT (why a strong feature can rank low in SHAP) ===")
    decomp = prevalence_lift_decomposition(dev)
    results["prevalence_lift_decomposition"] = decomp.to_dict(orient="records")
    print(f"(dev base cancer rate = {decomp.attrs['base_rate']})")
    print(decomp.to_string(index=False))

    # --- Q4: provenance ----------------------------------------------------
    print("\n=== PROVENANCE: is the target a deterministic rule? ===")
    rule = generative_rule_check(dev)
    results["generative_rule"] = rule
    print(f"Hypothesis: {rule['hypothesis']}")
    print(f"  accuracy, all dev rows        : {rule['accuracy_all_rows']}")
    print(f"  accuracy, complete cases only : {rule['accuracy_complete_cases']} "
          f"(n={rule['n_complete_cases']})")
    print("  cancer rate by number of conditions met:")
    for s, v in rule["cancer_rate_by_score"].items():
        print(f"     {s} conditions: n={v['n']:>5}  cancer rate={v['cancer_rate']:.4f}")

    # --- Encoding sanity check --------------------------------------------
    print("\n=== ENCODING CHECK: Mammogram_Result ordinal order ===")
    enc = encoding_order_check(dev)
    results["mammogram_encoding_check"] = enc
    print(json.dumps(enc, indent=2))

    os.makedirs("outputs/metrics", exist_ok=True)
    out_path = "outputs/metrics/investigations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    screen.to_csv("outputs/metrics/association_screen.csv", index=False)
    logger.info(f"Investigation results saved to {out_path} and association_screen.csv")
    print(f"\nSaved: {out_path}")
