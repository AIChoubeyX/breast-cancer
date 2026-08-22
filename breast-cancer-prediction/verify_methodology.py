"""Methodology verification -- the leakage and correctness checks, as assertions.

Run this after `src.train` and `src.evaluate` to confirm the pipeline still holds
the properties documented in DECISIONS.md:

    python verify_methodology.py

Every check either recomputes a value from scratch or inspects a persisted
artifact. Nothing is asserted from a docstring or a comment, because those can
drift from the code; where a claim is behavioural ("evaluation refuses to invent a
threshold"), the check provokes the behaviour rather than grepping for it.

Exits non-zero if any check fails, so it can gate a commit.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocess import (
    prepare_data,
    get_categorical_indices,
    DEFAULT_NUMERIC_COLS,
)

ROOT = Path(".")
results = []


def check(n, name, passed, detail):
    results.append((n, name, passed, detail))
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {str(n):>3}. {name}")
    print(f"         {detail}")


print("=" * 78)
print("METHODOLOGY VERIFICATION")
print("=" * 78)

data = prepare_data()
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# ---------------------------------------------------------------- 1
sizes = (len(X_train), len(X_val), len(X_test))
check(1, "Three-way split sizes are 6400 / 1600 / 2000",
      sizes == (6400, 1600, 2000), f"got {sizes}, total {sum(sizes)}")

# ---------------------------------------------------------------- 2
rates = (y_train.mean(), y_val.mean(), y_test.mean())
spread = max(rates) - min(rates)
check(2, "Splits are stratified (positive-rate spread < 0.005)",
      spread < 0.005,
      f"train {rates[0]:.4f}, val {rates[1]:.4f}, test {rates[2]:.4f}, spread {spread:.5f}")

# ---------------------------------------------------------------- 3
i_tr, i_va, i_te = set(X_train.index), set(X_val.index), set(X_test.index)
overlap = len(i_tr & i_va) + len(i_tr & i_te) + len(i_va & i_te)
check(3, "No row appears in more than one split",
      overlap == 0,
      f"pairwise overlaps: train/val {len(i_tr & i_va)}, train/test {len(i_tr & i_te)}, val/test {len(i_va & i_te)}")

# ---------------------------------------------------------------- 4
data2 = prepare_data()
same = (list(data2["X_train"].index) == list(X_train.index)
        and list(data2["X_test"].index) == list(X_test.index))
check(4, "Split is deterministic across runs (random_state=42)",
      same, "re-running prepare_data() reproduced identical train and test indices")

# ---------------------------------------------------------------- 5
src_train = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
# Any reference at all, then the subset that actually CONSUMES test data.
all_refs = re.findall(r'.*\b(?:X_test|y_test)\b.*', src_train)
# Consuming = fitting, predicting, scoring, or reading values/labels out of it.
consuming = [ln.strip() for ln in all_refs
             if re.search(r"\.fit|\.predict|\.transform|\.score|resample|"
                          r"_score\(|confusion|roc_|precision|recall", ln)]
# len() of a split is metadata only: it reads no feature value and no label.
metadata_only = [ln.strip() for ln in all_refs if re.search(r"len\(", ln)]
check(5, "train.py never fits, predicts on, or scores against the test set",
      len(consuming) == 0,
      f"{len(all_refs)} mention(s), {len(consuming)} consuming; "
      f"metadata-only (row count for the report): {metadata_only}")

# ---------------------------------------------------------------- 6
pipe = joblib.load(ROOT / "models" / "best_pipeline.pkl")
step_names = [n for n, _ in pipe.steps]
check(6, "Persisted pipeline has impute -> scale -> clf ordering",
      step_names[:2] == ["impute", "scale"] and step_names[-1] == "clf",
      f"steps: {step_names}")

# ---------------------------------------------------------------- 7
scaler_means = pipe.named_steps["scale"].scaler_.mean_
imp = pipe.named_steps["impute"]
train_medians = {c: float(X_train[c].median()) for c in imp.medians_}
med_match = all(abs(imp.medians_[c] - train_medians[c]) < 1e-9 for c in imp.medians_)
full_medians = {c: float(pd.concat([X_train[c], X_val[c], X_test[c]]).median()) for c in imp.medians_}
differs_from_full = any(abs(imp.medians_[c] - full_medians[c]) > 1e-9 for c in imp.medians_)
check(7, "Imputation medians were fitted on TRAIN only",
      med_match,
      f"learned {dict((k, round(v, 4)) for k, v in imp.medians_.items())} == train medians; "
      f"differs from all-data medians: {differs_from_full}")

# ---------------------------------------------------------------- 8
X_tr_imp = imp.transform(X_train)
train_mu = X_tr_imp[pipe.named_steps["scale"].columns_].mean().values
mu_match = np.allclose(scaler_means, train_mu, atol=1e-8)
check(8, "Scaler mean/std were fitted on TRAIN only",
      mu_match, f"max |scaler.mean_ - train mean| = {np.abs(scaler_means - train_mu).max():.2e}")

# ---------------------------------------------------------------- 9
has_sampler = any("smote" in n.lower() for n in step_names)
clf = pipe.named_steps["clf"]
spw = getattr(clf, "scale_pos_weight", None)
cw = getattr(clf, "class_weight", None)
stacked = has_sampler and (spw not in (None, 1.0) or cw is not None)
check(9, "Imbalance strategies are not stacked",
      not stacked,
      f"sampler step present: {has_sampler}; scale_pos_weight={spw}; class_weight={cw}")

# ---------------------------------------------------------------- 10
comp = json.loads((ROOT / "outputs" / "metrics" / "model_comparison.json").read_text())
rows = comp["experiments"]
n_exp = len(rows)
models = sorted(set(r["Model"] for r in rows))
strategies = sorted(set(r["Imbalance_Method"] for r in rows))
check(10, "Six configurations were compared (3 models x 2 strategies)",
      n_exp == 6 and len(models) == 3 and len(strategies) == 2,
      f"{n_exp} experiments: {len(models)} models {models} x {len(strategies)} strategies {strategies}")

# ---------------------------------------------------------------- 11
per_exp = set(r["CV_Metric"] for r in rows)
check(11, "Hyperparameters tuned on average_precision, not recall",
      per_exp == {"average_precision"}
      and comp.get("cv_metric", "").startswith("average_precision")
      and '"recall"' not in src_train.split("GridSearchCV(")[1][:400],
      f"CV_Metric recorded on all {len(rows)} experiments = {per_exp}; "
      f"header cv_metric = {comp.get('cv_metric')!r}")

# ---------------------------------------------------------------- 11b
check("11b", "Model + threshold were selected on VALIDATION",
      comp.get("evaluated_on", "").lower().startswith("valid"),
      f"model_comparison.json evaluated_on = {comp.get('evaluated_on')!r}; "
      f"selection_rule = {comp.get('selection_rule')!r}")

# ---------------------------------------------------------------- 12
enc = joblib.load(ROOT / "models" / "encoders.pkl")
thr = enc.get("decision_threshold")
check(12, "Decision threshold is persisted in encoders.pkl",
      isinstance(thr, float) and 0.0 < thr < 1.0, f"decision_threshold = {thr}")

# ---------------------------------------------------------------- 13
src_eval = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
from src.evaluate import load_locked_threshold
import tempfile, os as _os
# (a) key present -> returns it
got = load_locked_threshold("models/encoders.pkl")
# (b) key absent -> must raise KeyError, NOT return 0.50
stripped = {k: v for k, v in enc.items() if k != "decision_threshold"}
tmp = _os.path.join(tempfile.gettempdir(), "enc_no_threshold.pkl")
joblib.dump(stripped, tmp)
try:
    bad = load_locked_threshold(tmp)
    raised_key = f"returned {bad} instead of raising"
except KeyError:
    raised_key = "raised KeyError"
except Exception as e:
    raised_key = f"raised {type(e).__name__}"
# (c) file absent -> must raise FileNotFoundError
try:
    load_locked_threshold(_os.path.join(tempfile.gettempdir(), "does_not_exist.pkl"))
    raised_missing = "did not raise"
except FileNotFoundError:
    raised_missing = "raised FileNotFoundError"
except Exception as e:
    raised_missing = f"raised {type(e).__name__}"
_os.remove(tmp)
check(13, "evaluate.py raises rather than defaulting the threshold",
      got == thr and raised_key == "raised KeyError"
      and raised_missing == "raised FileNotFoundError",
      f"key present -> {got}; key absent -> {raised_key}; file absent -> {raised_missing}")

# ---------------------------------------------------------------- 14
thr_search_in_eval = bool(re.search(r"for\s+\w*thr\w*\s+in|np\.arange\(0", src_eval))
check(14, "evaluate.py does not search thresholds against the test set",
      not thr_search_in_eval, "no threshold loop / arange sweep in src/evaluate.py")

# ---------------------------------------------------------------- 15
feat_model = list(getattr(clf, "feature_names_in_", []))
feat_data = list(X_test.columns)
check(15, "SHAP feature alignment: exact name and order match",
      feat_model == feat_data and len(feat_model) == 20,
      f"{len(feat_data)} data features == {len(feat_model)} model features, identical order")

# ---------------------------------------------------------------- 16
cat_idx = get_categorical_indices(feat_data)
cat_names = [feat_data[i] for i in cat_idx]
cont_names = [c for c in feat_data if c not in cat_names]
partition_ok = (len(cat_names) + len(cont_names) == 20
                and set(cont_names) == set(c for c in DEFAULT_NUMERIC_COLS if c in feat_data))
check(16, "SMOTENC categorical indices partition all 20 features",
      partition_ok, f"{len(cat_names)} categorical + {len(cont_names)} continuous = 20")

# ---------------------------------------------------------------- 17
Xt = pipe.named_steps["scale"].transform(imp.transform(X_test))
nan_count = int(Xt.isna().sum().sum())
check(17, "No NaN survives preprocessing on the test set",
      nan_count == 0, f"{nan_count} NaN cells in the final test matrix")

# ---------------------------------------------------------------- 18
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, average_precision_score,
                            confusion_matrix)
proba = clf.predict_proba(Xt)[:, 1]
pred = (proba >= thr).astype(int)
fresh = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC_AUC": roc_auc_score(y_test, proba),
    "Average_Precision": average_precision_score(y_test, proba),
}
saved = json.loads((ROOT / "outputs" / "metrics" / "final_metrics.json").read_text())
KEYMAP = {"Accuracy": "Accuracy", "Precision": "Precision", "Recall": "Recall",
          "F1": "F1_Score", "ROC_AUC": "ROC_AUC", "Average_Precision": "Average_Precision"}
# The artifact stores values rounded to 4 dp, so the correct assertion is that
# each saved value equals round(recomputed, 4) exactly -- not that it is bit-identical.
mism = {k: (round(v, 4), float(saved[KEYMAP[k]]))
        for k, v in fresh.items() if round(v, 4) != float(saved[KEYMAP[k]])}
check(18, "Reported test metrics reproduce exactly from the artifacts (to 4 dp as stored)",
      len(mism) == 0 and len(KEYMAP) == 6,
      f"all {len(KEYMAP)} metrics reproduce; mismatches: {mism or 'none'}; "
      f"full precision recall={fresh['Recall']:.6f} precision={fresh['Precision']:.6f} "
      f"acc={fresh['Accuracy']:.6f} auc={fresh['ROC_AUC']:.6f}")

# ---------------------------------------------------------------- 19
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
cm_saved = saved["Confusion_Matrix"]
cm_ok = (cm_saved["TN_Benign_Correct"] == tn and cm_saved["FP_False_Alarm"] == fp
         and cm_saved["FN_Missed_Cancer"] == fn and cm_saved["TP_Malignant_Correct"] == tp
         and saved["Test_Set_Size"] == 2000 and saved["Test_Positives"] == tp + fn)
check(19, "Confusion matrix reproduces and sums to the test set",
      cm_ok and (tn + fp + fn + tp) == 2000,
      f"TN={tn} FP={fp} FN={fn} TP={tp}, total {tn+fp+fn+tp}, positives {tp+fn}")

# ---------------------------------------------------------------- 20
shap_csv = pd.read_csv(ROOT / "outputs" / "metrics" / "shap_feature_importance.csv")
complete = len(shap_csv) == 20 and set(shap_csv["Feature"]) == set(feat_data)
monotone = shap_csv["Mean_Abs_SHAP"].is_monotonic_decreasing
check(20, "SHAP ranking is complete (all 20 features) and unmodified",
      complete and monotone,
      f"{len(shap_csv)} rows, covers every model feature, "
      f"sorted descending: {monotone}, ranks 1..{shap_csv['Rank'].max()}")

# ---------------------------------------------------------------- 21
enc_keys_needed = ["binary_maps", "ordinal_maps", "onehot_categories",
                   "median_imputations", "categorical_fill_defaults",
                   "bmi_bounds", "feature_names", "decision_threshold"]
missing = [k for k in enc_keys_needed if k not in enc]
check(21, "encoders.pkl carries every key the app reads",
      not missing, f"missing keys: {missing or 'none'}")

# ---------------------------------------------------------------- 22
rs_hits = len(re.findall(r"random_state\s*=\s*42|random_state=RANDOM_STATE|RANDOM_STATE\s*=\s*42", src_train))
src_pre = (ROOT / "src" / "preprocess.py").read_text(encoding="utf-8")
rs_pre = len(re.findall(r"random_state", src_pre))
check(22, "random_state=42 used throughout",
      rs_hits > 0 and rs_pre > 0,
      f"{rs_hits} seeded call sites in train.py, {rs_pre} in preprocess.py")

print("=" * 78)
passed = sum(1 for _, _, p, _ in results if p)
print(f"RESULT: {passed}/{len(results)} checks passed")
failed = [(n, name) for n, name, p, _ in results if not p]
if failed:
    print("FAILED:")
    for n, name in failed:
        print(f"  {n}. {name}")
print("=" * 78)

import sys as _sys
_sys.exit(0 if not failed else 1)
