# Methodology Decisions

Every decision below is recorded with the evidence that drove it and the reasoning
that connects the two. Where the evidence was ambiguous, the more conservative
option was taken and the ambiguity is stated rather than resolved by assumption.

Reproduce the evidence with:

```bash
../venv/Scripts/python.exe -m src.investigate
```

Verify the methodology claims below still hold — 23 assertions, exits non-zero on
failure:

```bash
../venv/Scripts/python.exe verify_methodology.py
```

---

## 0. The finding that governs how every other number here should be read

**The target column is a deterministic function of eight features.** It is not a
noisy probabilistic outcome.

```
score = [Age >= 50] + [BMI >= 30] + [Tumor_Size_cm >= 3.0] + [Family_History == Yes]
      + [Smoking == Yes] + [Genetic_Mutation == Positive]
      + [Lymph_Node_Involvement == Yes] + [Mammogram_Result != Normal]

Cancer = 1  if and only if  score >= 4
```

Measured on the dev split (8000 rows; the test set was not read):

| Number of conditions met | Rows | Cancer rate |
|---:|---:|---:|
| 0 | 429 | 0.0000 |
| 1 | 1527 | 0.0000 |
| 2 | 2390 | 0.0038 |
| 3 | 2117 | 0.0104 |
| **4** | **1072** | **0.9795** |
| 5 | 389 | 1.0000 |
| 6 | 71 | 1.0000 |
| 7 | 4 | 1.0000 |
| 8 | 1 | 1.0000 |

Rule accuracy: **0.993375** on all dev rows, **0.997167** on complete cases
(n=7766). The residual disagreement is confined to rows with missing values in
the eight columns.

**Three consequences, all of which constrain what this project can claim:**

1. **The high test scores are not evidence of a good model.** Accuracy 0.9945 and
   ROC-AUC 0.9982 are what any competent learner achieves against a rule with
   almost no irreducible noise. They measure the dataset, not the modelling.

2. **They are not evidence of leakage either.** This was the initial suspicion and
   it is wrong. No single column approaches the label once `Biopsy_Result` and
   `Cancer_Stage` are dropped — the highest surviving univariate AUC is 0.6805.
   The separability is distributed across eight legitimate risk factors by
   construction, so no feature can be dropped to remove it.

3. **The SHAP ranking cannot be read clinically.** The generator weights its eight
   conditions equally. Any ordering among them reflects how often each condition
   fires in the cohort, not how much any of them matters medically.

---

## 1. Feature decisions

### 1.1 `Patient_ID` — REMOVED

An arbitrary row identifier. It carries no clinical information and inviting a
model to memorise it invites overfitting to nothing.

### 1.2 `Biopsy_Result` — REMOVED (target proxy)

`Biopsy_Result == Malignant` maps onto `Cancer == 1` with **zero exceptions**.
On the dev split: 1546 Malignant rows, 1546 cancers, **0 counter-examples**;
Cramér's V = 1.000000; univariate AUC = 1.000000; χ² = 8000.0, exactly equal to n
— the algebraic signature of a deterministic relationship.

This column *is* the label under another name. It is also a post-diagnostic
observation: a biopsy result exists only after cancer has been investigated, so
using it to predict cancer inverts the causal order.

### 1.3 `Cancer_Stage` — REMOVED (target proxy)

`Cancer_Stage == "No Cancer"` covers exactly the 8067 negatives in the full
dataset. Same reasoning as `Biopsy_Result`; staging is downstream of diagnosis.

### 1.4 `Lymph_Node_Involvement` — **RETAINED**

This was flagged as a possible proxy on the reasoning that nodal status is
normally established by biopsy or surgical staging, i.e. after a malignancy
workup has begun. That concern is clinically sound and was worth raising. The
data refutes it.

The decisive test is the off-diagonal count, not the correlation strength. A
genuine risk factor has many patients who carry it without the disease; a label
proxy has none.

| Evidence (dev split) | `Lymph_Node_Involvement` | `Biopsy_Result` (removed) |
|---|---:|---:|
| Patients at highest-risk level | 1635 | 1546 |
| **Cancer-free patients at that level** | **983** | **0** |
| Share of that level that is cancer-free | **60.12%** | 0.00% |
| Share of all cancers captured | 42.17% | 100.00% |
| Cramér's V | 0.263594 | 1.000000 |
| Univariate AUC | 0.634712 | 1.000000 |
| χ² (n = 8000) | 556.79 | **8000.0 = n** |
| Verdict | risk factor | deterministic proxy |

983 cancer-free patients have positive nodal status, and the feature misses 58% of
cancers. Both facts are incompatible with it restating the diagnosis. It behaves
like `Genetic_Mutation` (V = 0.250556 on the full data, a difference of 0.000320 —
statistically indistinguishable), not like `Biopsy_Result`.

**A consistency argument matters here too.** `Tumor_Size_cm` has a *stronger*
clinical-timing objection than nodal status: a measured tumour size presupposes a
detected tumour, and its univariate AUC (0.664107 on full data) is higher.
Excluding lymph node status on timing grounds while retaining tumour size would
be incoherent. Either both belong at this prediction point or neither does.

### 1.5 `Mammogram_Result` — **RETAINED**

Ranked #1 by mean|SHAP| (0.1429 of total attribution) and the strongest surviving
univariate feature (risk-ordered AUC 0.680459). Retained on availability grounds:
the app's own input form asks for it, and its batch mode already requires it, so
the application's established prediction point is post-imaging.

**Documented caveat on its encoding.** The configured ordinal order is
`Normal < Suspicious < Abnormal`, but observed risk runs
`Normal (0.1097) < Abnormal (0.3157) < Suspicious (0.3760)`. The configured order
is therefore **not monotone in risk** — it places Abnormal above Suspicious when
Suspicious carries slightly higher risk.

This is harmless *for this model* and worth stating plainly rather than quietly
fixing: the generator only uses the binary `[!= Normal]`, so the Suspicious /
Abnormal distinction is noise, and a tree model splits at a threshold that
isolates `Normal` regardless of how the other two are ordered. It would matter for
a linear model, where the codes are multiplied by a single coefficient. For
reference, alphabetical ordering would be actively harmful (AUC 0.549819 versus
0.680459 risk-ordered) because it puts the lowest-risk level in the middle.

### 1.6 The eleven noise features — **RETAINED**

`Gender`, `Alcohol_Consumption`, `Physical_Activity`, `Hormone_Therapy`,
`Menopause_Status`, `Diabetes`, `Breastfeeding_History`, `Blood_Pressure`,
`Cholesterol`, `Exercise_Days_Per_Week`, `Annual_Income_USD`.

All have association ≤ 0.017 with the target and p > 0.10; Bergsma's bias
correction drives five of them to exactly zero. They are noise by construction.

They were retained rather than pruned because **the trained model already
discards them**, which is a more honest demonstration than removing them by hand:
eight of them receive mean|SHAP| of *exactly* 0.000000, and the remaining three
sit at 0.18% of total attribution or below. Removing them would also break the
app's input form for no measurable gain.

### 1.7 `Tumor_Size_cm == 0.0` (13 rows) — **LEFT UNCHANGED**

A 0 cm tumour is physiologically impossible, and non-zero values start at 0.01,
which makes 0.0 look like a sentinel. But 11 of the 13 rows are `Cancer == 0`, so
0.0 plausibly encodes "no mass detected" — legitimate information rather than a
missing value.

The provenance is genuinely ambiguous and the dataset provides no way to settle
it. Coercing these to NaN would impute the training median (2.5 cm), inserting a
tumour into patients who may have none; that is a larger and less reversible
distortion than leaving 13 of 10000 rows (0.13%) alone. **Recorded as a limitation
rather than resolved by assumption.**

---

## 2. Leakage and methodology decisions

### 2.1 Preprocessing split into deterministic and learned

The governing rule: a transform may run before the split only if its behaviour is
fixed in advance and identical for every row. Anything that computes a statistic
*from* the data must be fitted inside the training folds.

| Transform | Class | When |
|---|---|---|
| Constant categorical fills (`No`, `Moderate`) | deterministic | before split |
| Fixed clinical BMI bounds [15, 50] | deterministic | before split |
| Binary / ordinal / one-hot maps | deterministic | before split |
| **Median imputation** | **learned** | inside training folds |
| **StandardScaler mean & std** | **learned** | inside training folds |

The previous pipeline imputed medians over the full dataset before splitting, so
every median carried information from rows that later became test rows.

Fixed bounds are not percentile clipping. `[15, 50]` comes from physiology, not
from this dataset's distribution, so it leaks nothing.

### 2.2 SMOTENC instead of SMOTE

13 of 20 columns are binary, ordinal or one-hot. Plain SMOTE interpolates linearly
between neighbours, which generates `Family_History = 0.63` and
`Mammogram_Result = 1.37` — states no patient can occupy. Training on those
teaches the model to interpret a scale that does not exist at inference time.

SMOTENC interpolates continuous columns and takes the majority level among
neighbours for categorical ones. Its categorical columns are identified **by
integer index**, so the indices are derived programmatically from the live feature
list and printed for verification; a hard-coded index list would silently corrupt
resampling the moment column order changed. `get_categorical_indices` raises if the
categorical and continuous sets fail to cover every feature.

### 2.3 Resampling moved inside the cross-validation pipeline

Previously SMOTE ran once over the whole training set, and the resampled matrix
was then handed to `GridSearchCV`. Synthetic rows interpolated from a patient in
fold 3 therefore appeared in folds 1, 2, 4 and 5 — so every fold was scored partly
on information derived from its own held-out data, and CV scores were inflated.

Resampling now sits inside an `imblearn.Pipeline`, which applies samplers during
`fit` and skips them during `predict`. Each fold resamples only its own training
portion, and validation folds keep their natural class balance.

### 2.4 Imbalance strategies compared, not stacked

The previous pipeline applied SMOTE **and** `class_weight="balanced"` **and**
`scale_pos_weight` simultaneously. Each of these reweights the minority class, so
together they multiply: the minority class was effectively upweighted several
times over by an amount nobody had computed.

The two mechanisms are now mutually exclusive alternatives:

- **Experiment A — `class_weight`:** cost-sensitive learning, no resampling.
- **Experiment B — `smotenc`:** resampling, no additional weighting.

**Result: cost-sensitive weighting won in all three pairings, by 5-fold CV average
precision on the training split.**

| Model | A: class_weight | B: smotenc | Δ |
|---|---:|---:|---:|
| XGBoost | **0.9969** ± 0.0030 | 0.9848 ± 0.0044 | +0.0121 |
| RandomForest | **0.9882** ± 0.0041 | 0.9620 ± 0.0059 | +0.0262 |
| LogisticRegression | **0.8989** ± 0.0093 | 0.8794 ± 0.0122 | +0.0195 |

Consistent direction across three model families, with gaps several times the CV
standard deviation. Synthetic minority rows did not help here — plausibly because
a threshold rule has sharp decision boundaries, and interpolating across them
manufactures rows whose labels the rule would not assign. SMOTENC also cost about
3× the training time.

### 2.5 Hyperparameters tuned on average precision, not recall

The previous grid search optimised `recall`, which is degenerate as a selection
criterion: predicting "malignant" for every patient scores recall 1.0 at zero
clinical value. The search was rewarded for choosing the most permissive model.

Model quality and operating point are now chosen separately. `GridSearchCV` scores
on **average precision** — threshold-free and focused on the positive class — and
the decision threshold is chosen afterwards, on validation, to hit the recall
target.

### 2.6 A real validation set now exists

Previously there was no validation set. Models were "validated" on the same
training observations they were fitted on (the code comment conceded it:
*"Evaluate on original (non-SMOTE) training set as quick validation"*).

| Split | Share | Rows | Positive rate |
|---|---:|---:|---:|
| Train | 64% | 6400 | 0.1933 |
| Validation | 16% | 1600 | 0.1931 |
| Test | 20% | 2000 | 0.1935 |

Stratified, `random_state=42`. Both splits derive from one seeded two-stage split,
so training and evaluation reconstruct identical row sets without persisting them.

### 2.7 Threshold selected on validation, then locked

This was the most consequential leak. The previous `evaluate.py` searched 81
candidate thresholds **against `y_test`** and reported the best one. The published
recall was therefore the maximum achievable on the test set with hindsight, not an
estimate of field performance. (The search also used `r_t >= best_r` over ascending
candidates, biasing it toward the highest qualifying threshold.)

Threshold selection now happens in `src/train.py` on validation data, over
`{0.50, 0.45, 0.40, 0.35, 0.30}`, under a rule fixed in advance:

1. Keep candidates with validation precision ≥ 0.70.
2. Among those, take the highest recall.
3. Break ties on F1, then on the higher threshold.
4. If none clears the precision floor, fall back to maximum F1 **and warn**.

The chosen value is persisted to `models/encoders.pkl` as `decision_threshold`.
`src/evaluate.py` **raises** if that key is absent rather than defaulting to 0.50,
because a silent default would quietly reintroduce exactly the bug this separation
exists to prevent.

Selected threshold: **0.50**. Recall was flat at 0.9903 across every candidate
from 0.30 to 0.50 while precision fell monotonically from 0.9808 to 0.9027, so the
rule selected the highest-precision member of a recall-tied set. Flat recall is
itself a signature of the deterministic rule: predicted probabilities sit near 0
and 1, so moving the threshold reclassifies almost nobody.

Rule 4 did fire for both LogisticRegression configurations, which could not reach
precision 0.70 at any candidate. That is logged as a warning rather than hidden.

### 2.8 Model selected on validation, with a stated caveat

Selection ranks on validation recall, then F1, then ROC-AUC. Accuracy is
deliberately not a criterion: at 19.3% prevalence, predicting "benign" for
everyone scores 80.7%.

**Caveat, stated because it is a real if minor weakness:** the validation set is
used both to choose each configuration's threshold and to rank configurations. The
winner is therefore mildly optimistic on validation. This is the standard use of a
validation set and the test set remains uncontaminated, so the reported test
metrics are unaffected — but the validation figures should be read as selection
scores, not as unbiased estimates.

### 2.9 Final model fitted on TRAIN only

The final model is fitted on the training split alone, not train + validation.
Refitting on both would have meant the locked threshold was chosen on data the
final model had since trained on, making the operating point no longer
out-of-sample. The cost is 1600 unused training rows; the benefit is that the
threshold remains honest.

### 2.10 DataFrames preserved end to end instead of `ColumnTransformer`

`ColumnTransformer` reorders columns — it emits transformed blocks in the order
its transformers are declared, then appends passthrough columns. Two things here
depend on column order: SMOTENC's integer categorical indices, and every SHAP
feature label. A silent reordering would corrupt both while raising no error.

`MedianImputerDF` and `StandardScalerDF` therefore accept and return DataFrames
with column order intact, so the same names reach the classifier, SMOTENC and SHAP.

### 2.11 SHAP alignment verified by assertion, not by inspection

The suspicion that SHAP values were computed on one array and labelled with a
different column order was worth raising — it produces exactly this kind of
implausible ranking, and it is invisible without an explicit check, because the
numbers stay internally valid while every label is wrong.

**It was not the cause.** In the previous code, `feature_names` derived from the
same `X_test_scaled` DataFrame that was handed to the explainer, so labels were
already aligned. To make that guarantee explicit rather than incidental,
`verify_feature_alignment()` now compares `list(X_test_final.columns)` against
`model.feature_names_in_` element by element and **raises** on mismatch.

Verified: 20 data features, 20 model features, exact order match, source
`estimator.feature_names_in_`.

---

## 3. Why the SHAP ranking looks the way it does

The complete ranking is in `outputs/metrics/shap_feature_importance.csv`. It was
not reordered, filtered or adjusted.

| Rank | Feature | mean\|SHAP\| | % of total |
|---:|---|---:|---:|
| 1 | Mammogram_Result | 2.001160 | 14.29 |
| 2 | Family_History | 1.860532 | 13.29 |
| 3 | Smoking | 1.846722 | 13.19 |
| 4 | Age | 1.794174 | 12.81 |
| 5 | Tumor_Size_cm | 1.779964 | 12.71 |
| 6 | Lymph_Node_Involvement | 1.701512 | 12.15 |
| 7 | BMI | 1.650970 | 11.79 |
| 8 | Genetic_Mutation | 1.327843 | 9.48 |
| 9 | Cholesterol | 0.024958 | 0.18 |
| 10 | Exercise_Days_Per_Week | 0.010360 | 0.07 |
| 11 | Annual_Income_USD | 0.002167 | 0.02 |
| 12 | Blood_Pressure | 0.000643 | 0.00 |
| 13–20 | Gender, Alcohol_Consumption, Physical_Activity, Hormone_Therapy, Menopause_Status, Diabetes, Breastfeeding_History ×2 | **0.000000** | 0.00 |

**The structure is the finding.** The top 8 are exactly the 8 features in the
generative rule, spanning a narrow 9.48%–12.81%–14.29% band. Rank 8 to rank 9 is a
**53× cliff** (1.327843 → 0.024958). Ranks 13–20 are *exactly* zero: the model
independently discovered which columns are noise.

### 3.1 Why `Smoking` (#3) outranks `Age` (#4)

The gap is **0.052548**, or 2.9% relative — both sit inside the equal-weight band.
This is not a bug, and not a labelling error (alignment verified above). It is
faithful to the data, and the two standard importance metrics disagree about it:

| Metric | Age | Smoking | Winner |
|---|---:|---:|---|
| Univariate AUC | 0.665102 | 0.651261 | Age |
| Risk ratio | 3.084507 | 2.914299 | Age |
| **Multivariate ablation (AUC drop)** | 0.023778 | **0.035618** | **Smoking** |
| mean\|SHAP\| | 1.794174 | **1.846722** | **Smoking** |

SHAP reports marginal contribution in the presence of other features, which tracks
the ablation result, not the univariate one. `Age >= 50` overlaps with other
conditions; `Smoking == Yes` is more nearly independent, so removing it costs more.

**The clinical objection was correct and remains correct.** Smoking's real
association with breast cancer is weak and still debated; it does not belong
alongside family history and age. The reason it appears here is that the generator
assigned `[Smoking == Yes]` and `[Age >= 50]` **equal weight** as two of eight
interchangeable conditions. All eight risk ratios fall in the narrow band
2.718–3.189, which is what equal weighting looks like from the outside.

This is evidence about the data generator, not about breast cancer. It must not be
presented as a clinical finding.

### 3.2 Why `Genetic_Mutation` ranks 8th and `Tumor_Size_cm` 5th

Both are in the top 8 — that is, both are among the features that carry all of the
signal. Neither absence from the top 5 indicates a defect.

`Genetic_Mutation` ranks last of the eight because **mean|SHAP| blends effect size
with reach**, and it has the strongest lift of any feature but the smallest
footprint:

| Feature | Risk ratio | Prevalence | Share of all cancers |
|---|---:|---:|---:|
| Mammogram_Result != Normal | 3.188541 | 0.348 | 63.0% |
| Age >= 50 | 3.084507 | 0.535 | **78.0%** |
| **Genetic_Mutation == Positive** | **2.925347** | **0.145** | **33.2%** |
| Smoking == Yes | 2.914299 | 0.254 | 49.8% |
| Lymph_Node_Involvement == Yes | 2.839166 | 0.204 | 42.2% |
| Family_History == Yes | 2.718200 | 0.302 | 54.1% |

`Genetic_Mutation` is nearly tied with `Age` on lift but applies to a quarter as
many patients, so its cohort-averaged attribution is diluted. **A low mean|SHAP|
rank is not evidence that a feature is unimportant for the patients it applies
to** — for a mutation-positive patient it is among the strongest signals available.

`Tumor_Size_cm` has the highest univariate AUC of any retained feature (0.664107)
yet ranks 5th, because the generator uses only `[>= 3.0 cm]`. It is a step
function: a 9 cm tumour contributes no more than a 3.1 cm tumour, so the extra
range carries no information for the model to use.

---

## 4. Limitations

1. **The dataset is synthetic, and this is not a marginal caveat.** The label is a
   deterministic 4-of-8 threshold rule (99.72% on complete cases). No clinical
   conclusion may be drawn from any number in this project.

2. **No clinical validity is claimed.** This is a decision-support prototype and a
   methodology demonstration. It is not a diagnostic instrument and must not
   inform patient care.

3. **Feature importances are not causal.** SHAP describes how *this model* uses
   *these columns*. On this data it recovers the generator's equal weighting, not
   biological mechanism.

4. **Reported metrics do not transfer.** Accuracy 0.9945 and ROC-AUC 0.9982
   describe performance against a near-noiseless rule. Real cohorts contain
   irreducible uncertainty; expect substantially lower figures.

5. **Semantic contradictions remain in the data**, left in place deliberately
   because silently "fixing" generated data would misrepresent it:
   - 723 males, all assigned a `Menopause_Status`, and 556 assigned a real
     `Breastfeeding_History` rather than "Not Applicable".
   - `Menopause_Status` is independent of `Age` — 1739 post-menopausal rows under
     40, 1696 pre-menopausal rows over 60.
   - `Physical_Activity` contradicts `Exercise_Days_Per_Week`: the "Low" group
     exercises *more* (3.587 days) than the "High" group (3.432).
   - 510 rows with `Cancer == 0` and `Tumor_Size_cm > 5`; 1235 with `Cancer == 0`
     and positive nodal status.
   - `Blood_Pressure` is systolic-only with zero signal (r = −0.0038).
   - `Age`, `Blood_Pressure`, `Cholesterol`, `Annual_Income_USD` and
     `Exercise_Days_Per_Week` are uniformly, not normally, distributed.

6. **`data/breast_cancer_cleaned.csv` is not cleaned.** It is the raw file minus
   three columns: 0 cells changed, 0 nulls filled, 0 rows removed. All imputation
   and encoding happens in code. The filename is misleading.

7. **Missingness is weakly informative.** The 300 incomplete rows have a 0.1533
   cancer rate against 0.1945 for complete rows, so dropping them would discard 46
   positives and shift the class balance. They are imputed, not dropped.

8. **Validation figures are selection scores.** See §2.8 — validation is used for
   both threshold and model choice, so its metrics are mildly optimistic. Only the
   test metrics are unbiased.

9. **The prediction point was inferred, not specified.** No written specification
   of what is known at prediction time exists in this repository. It was inferred
   from the app's input form and batch requirements as *after imaging and physical
   examination, before biopsy and staging*. `Mammogram_Result`, `Tumor_Size_cm` and
   `Lymph_Node_Involvement` are retained on that basis. **If the intended
   deployment is pre-imaging population screening, all three become unavailable and
   the model must be retrained without them.** That is a scoping decision, not a
   methodological one, and it is outside what the evidence here can settle.

10. **`Tumor_Size_cm == 0.0` (13 rows) is unresolved** — see §1.7.
