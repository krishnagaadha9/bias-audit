# Governance Report: Fairness Audit of a Credit-Scoring Model

**System audited:** Logistic regression credit-risk classifier trained on the German Credit dataset (1,000 applicants, UCI/StatLog, 1994)
**Audit date:** 2026-08-12
**Audit artifacts:** `bias_audit.ipynb`, `bias_audit_results.xlsx`, `bias_audit_results_both.xlsx`, `bias_audit_stability_summary.xlsx`

---

## 1. What was audited, and why

This report covers a fairness audit of a model that predicts whether a loan applicant is a good or bad credit risk. Credit scoring was selected as the audit subject because it is a decision that directly and materially affects a person's access to financial services, and because — as shown below — it falls under a specific high-risk category in EU AI Act.

The audit asked three questions:

1. Does the model's error rate differ across protected groups (sex, age, foreign-worker status), and by how much?
2. Does removing protected attributes from the model's inputs ("fairness through unawareness") actually reduce that disparity, or do proxy features reconstruct it?
3. Are the disparities we observe stable findings, or could they be sampling noise given the dataset's size?

Sex and age were reconstructed from the raw `personal_status_sex` and `age` columns (age split at 25, following prior published work on this dataset). Two model variants were trained and compared: `DROP_PROTECTED=True` (sex, age, and foreign-worker status excluded from model inputs) and `DROP_PROTECTED=False` (included). All group-level metrics were computed on a held-out test set of 300 applicants (stratified 70/30 split, `random_state=42`), and re-checked across 19 random seeds plus 1,000-iteration bootstrap resampling to separate real effects from noise.

## 2. Key findings

### 2.1 The disparity exists before any model is trained

In the raw labels, before any model touches the data: women are labelled good credit 64.8% of the time versus 72.3% for men (n=310 / n=690) — a 7.5-point gap. Applicants under 25 are labelled good credit 59.1% of the time versus 71.9% for applicants 25 and over (n=149 / n=851) — a 12.8-point gap. Any model trained on these labels starts from an unequal baseline; it is not the sole origin of the disparities that follow.

### 2.2 Model performance is barely above the naive baseline

With protected attributes visible (`DROP_PROTECTED=False`), the model reaches 73.7% accuracy on the 300-person test set, versus 70.0% for a rule that approves every applicant — a 3.7-point improvement. Recall on bad-credit applicants is 49%: the model misses just over half of the genuinely bad risks it is meant to catch.

### 2.3 Selection-rate and outcome gaps by sex (visible-attribute model)

| Metric | Female (n=96) | Male (n=204) | Gap |
|---|---|---|---|
| Selection rate (approval rate) | 66.7% | 77.9% | 11.2 pts |
| True positive rate (approved when creditworthy) | 77.8% | 87.1% | 9.3 pts |
| False positive rate (approved when not creditworthy) | 45.5% | 54.4% | 8.9 pts |

The disparate impact ratio (female rate ÷ male rate) is **0.855** — above the commonly cited 0.80 "four-fifths" guideline, though that guideline originates in U.S. employment law, not lending regulation, and is cited here only as a reference point.

### 2.4 Age and nationality

The age gap is the largest single disparity measured: 66.0% selection rate for under-25 applicants (n=47) versus 75.9% for 25-and-over (n=253), a 9.9-point gap — though this is smaller than the 12.8-point gap already present in the raw labels, meaning the model did not amplify this disparity. The foreign-worker comparison (13 non-foreign-worker applicants in the test set versus 287 foreign-worker applicants) is too small a sample to support any conclusion and is reported here only to flag that it cannot currently be audited with this dataset.

### 2.5 Hiding protected attributes did not remove the disparity

Comparing the two model variants directly:

| | Accuracy | Disparate impact ratio (sex) |
|---|---|---|
| Protected attributes hidden | 73.3% | 0.915 |
| Protected attributes visible | 73.7% | 0.855 |

Hiding sex, age, and foreign-worker status narrowed the sex disparity but did not eliminate it. Features such as `housing_rent`, `employment_duration`, `job`, and `property` correlate with sex and age and act as proxies, letting the model partially reconstruct the excluded information. "Fairness through unawareness" is not a sufficient fairness intervention on its own for this system.

### 2.6 Which gaps are real versus noise

Bootstrap resampling (1,000 iterations) on the sex selection-rate gap shows that with protected attributes **visible**, the gap is statistically robust: mean 0.116, 95% CI [0.011, 0.226], with only a 1.7% chance the true gap is zero or negative. With attributes **hidden**, the same gap is not statistically distinguishable from noise: mean 0.065, 95% CI [-0.045, 0.175], 11.5% chance the true gap is zero or negative. The age gap is uncertain in both settings — its confidence interval includes zero regardless of whether protected attributes are visible. **The finding that survives scrutiny is specific: making sex visible to the model produces a sex disparity that is statistically robust, not a coincidence of one random split.**

### 2.7 Per-applicant explainability (SHAP)

Feature-level coefficients describe the model's average behavior but not what drove any individual decision. SHAP (Shapley Additive exPlanations) values were computed for every applicant in the test set using `shap.LinearExplainer`, decomposing each prediction into an exact per-feature, per-person contribution. This is the level of explanation an adverse-action notice or a regulator's individual-decision review would require — "your application status and loan amount contributed X and Y to this specific denial" — rather than a population-level coefficient that does not speak to any one person's outcome.

## 3. Regulatory classification: EU AI Act

Under the EU AI Act (Regulation (EU) 2024/1689), a system of this kind — used to evaluate the creditworthiness of natural persons or establish their credit score — is classified as **high-risk AI** under **Article 6(2)**, read together with **Annex III, point 5(b)**. Article 6 sets out the classification rule (systems listed in Annex III are high-risk unless a narrow profiling-immaterial exception applies), and Annex III point 5(b) names creditworthiness evaluation and credit scoring explicitly, with a carve-out only for systems used solely to detect financial fraud.

High-risk classification carries specific, non-optional obligations that a production version of this system would need to satisfy and that this audit only partially demonstrates:

- **Risk management system** (Article 9) — continuous identification and mitigation of the risks this report surfaces, not a one-time audit.
- **Data governance** (Article 10) — this dataset's known deficiencies (Section 4) would need remediation, not disclosure alone, before deployment.
- **Technical documentation and record-keeping** (Articles 11–12) — this report and its underlying notebook are a starting point for that documentation, not a substitute for it.
- **Transparency to affected persons** (Article 13) — applicants are entitled to know an AI system is involved in the decision.
- **Human oversight** (Article 14) — a human must be able to meaningfully review and override individual decisions; a 49% recall on bad-credit applicants (Section 2.2) is exactly the kind of error rate that makes this oversight requirement substantive rather than procedural.
- **Accuracy, robustness, and cybersecurity** (Article 15).

This system, as it exists in this repository, is a research and audit artifact, not a deployed decision system, and does not currently meet these obligations. This section exists to make clear what would be required if it — or a system like it — were put into production.

## 4. Policy recommendations

**1. Do not deploy this model, or any credit model, on labels inherited from historical human decisions without an independent audit of the labels themselves.** The base-rate gaps in Section 2.1 exist before any model is trained, and Section 2.6 shows the model reproduces a statistically robust version of the sex gap when it can see sex directly. No fairness metric computed against these labels can detect whether the original lending decisions were themselves discriminatory — the labels are the yardstick and the thing being measured at once. Recommendation: commission a review of the original decision process (or, for a live system, an outcomes-based review using repayment data rather than approval data) before treating this dataset's labels as ground truth for any deployed model.

**2. Do not rely on removing protected attributes as a fairness control.** Section 2.5 shows this measurably fails to remove the disparity, only narrows it, because ordinary features (housing, employment duration, job, property) act as proxies. Recommendation: if protected-attribute-blind modeling is used, it must be paired with proxy-feature detection (e.g., measuring each feature's correlation with protected attributes) and post-hoc outcome audits like the one in this report — not treated as a fairness guarantee in itself.

**3. Set a minimum reportable subgroup size and a mandatory uncertainty disclosure for any fairness metric before it is acted on.** The foreign-worker comparison in this audit (13 versus 287 applicants) produced a selection-rate gap that looks dramatic but is not statistically meaningful at that sample size, and the age gap's bootstrap interval spans zero in both model variants. Recommendation: adopt a policy — e.g., no subgroup fairness metric is reported or acted on below n=30, and every reported gap carries a confidence interval — so that governance decisions are not made on noise, and so that data-limited groups are flagged as *unauditable* rather than silently omitted or falsely cleared.

## 5. Limitations of this audit

- The dataset is 1,000 applicants from 1994 Germany; lending practices, law, and the underlying population have changed substantially since.
- Sex is recorded as a binary category in the source data; this audit could not assess disparities for anyone outside that binary.
- The foreign-worker subgroup (n=37 in the full dataset, n=13 in the test set) is too small to audit with this data; this is a data-collection gap, not a finding of fairness.
- This audit measures outcome disparities against the dataset's own labels. It cannot detect bias that is already embedded in those labels (Section 2.1, Section 4).
