# German Credit Bias Audit

## Why This Matters

Credit scoring is one of the first places algorithmic decision-making meets binding fairness law. Under the EU AI Act, systems that evaluate creditworthiness are legally classified as **high-risk AI** — not a best-practice suggestion, a regulatory category with mandatory obligations attached.

This repository builds a credit-risk model end-to-end and then audits it the way that classification demands: quantify group-level disparities with named fairness definitions, test whether a common but insufficient fix (hiding protected attributes) actually works, check whether the disparities found are statistically real or sampling noise, and produce per-decision explainability rather than only model-level coefficients. The full analysis is in `bias_audit.ipynb`; the written findings are in `reports/governance_report.md`.

## EU AI Act Classification

This system falls under **Article 6(2)**, read together with **Annex III, point 5(b)**, which names creditworthiness evaluation and credit scoring explicitly as high-risk (with a carve-out only for fraud-detection systems). That classification brings a defined set of obligations: a risk management system (Article 9), data governance (Article 10), technical documentation (Articles 11–12), transparency to affected individuals (Article 13), human oversight (Article 14), and accuracy/robustness requirements (Article 15).

This project is a research and audit artifact, not a deployed system, so it does not claim to satisfy those obligations — but it is structured around them: `reports/governance_report.md` maps each finding back to the obligation it informs.

## Contents

- `bias_audit.ipynb` — main notebook with data loading, model training, fairness metrics, SHAP explainability, and audit narrative.
- `german_credit_raw.csv` — raw dataset used for the audit.
- `bias_audit_results.xlsx` — saved audit results from the notebook.
- `bias_audit_results_both.xlsx` — comparison results for both hidden and visible protected-attribute runs.
- `bias_audit_stability_summary.xlsx` — stability and bootstrap analysis for random-state variation and noise estimation.
- `reports/governance_report.md` — written governance report: findings, EU AI Act high-risk classification, and policy recommendations.

## What this project shows

- Logistic regression is used to classify credit risk.
- Protected attributes include `sex`, `age`, and `foreign_worker`.
- The notebook compares two settings:
  - `DROP_PROTECTED=True` — protected attributes removed from model inputs.
  - `DROP_PROTECTED=False` — protected attributes included in the model.
- The audit measures selection-rate gaps, disparate impact, and stability across random seeds.
- Bootstrap analysis estimates whether observed gaps are stable or likely due to sampling noise.
- SHAP (`shap.LinearExplainer`) decomposes each individual prediction into per-feature contributions, on top of the model-level coefficients.

## Findings

| Metric | Result |
|---|---|
| Model accuracy vs. naive baseline | 73.7% vs. 70.0% ("approve everyone") |
| Selection rate — female vs. male | 66.7% vs. 77.9% (11.2-pt gap, n=96/204) |
| True positive rate — female vs. male | 77.8% vs. 87.1% (9.3-pt gap) |
| Disparate impact ratio (sex) | 0.855 — above the commonly cited 0.80 reference threshold |
| Selection rate — under-25 vs. 25-and-over | 66.0% vs. 75.9% (9.9-pt gap, n=47/253) |
| Effect of hiding protected attributes | Narrows the sex disparity (DI ratio 0.855 → 0.915) but does not eliminate it — proxy features (housing, employment duration, job, property) partially reconstruct the hidden signal |
| Statistical robustness (1,000-iteration bootstrap) | Sex gap is statistically robust when attributes are visible (95% CI [0.011, 0.226]); not distinguishable from noise when hidden (95% CI [-0.045, 0.175]) |
| Age gap robustness | Uncertain in both settings — confidence interval includes zero regardless of `DROP_PROTECTED` |

Full numbers, per-group breakdowns, and interpretation are in `reports/governance_report.md`.

## How to run

1. Install dependencies:
   ```bash
   python3 -m pip install pandas scikit-learn openpyxl shap
   ```
2. Open `bias_audit.ipynb` in Jupyter or a compatible notebook viewer.
3. Run the notebook cells in order.
4. Use the `DROP_PROTECTED` switch to compare audit modes.

## Notes and limitations

- The dataset is from 1994 Germany, so it reflects historical credit decisions.
- Some subgroups are small, making fairness ratios noisy — the foreign-worker comparison (13 vs. 287 in the test set) is reported but flagged as unauditable at that sample size, not treated as a finding.
- Sex categories are binary and incomplete.
- The labels are bank decisions, which may themselves encode unfairness. No metric in this audit can detect bias that is already embedded in the labels it measures against.

## Context

This project was built to demonstrate applied skills at the intersection of machine learning and AI governance:

- **Fairness/bias auditing** — multiple fairness definitions (demographic parity, equal opportunity, equalized odds), applied and explicitly compared rather than collapsed into one score.
- **Regulatory literacy** — mapping a technical system to its actual EU AI Act classification and obligations, not just citing the regulation in passing.
- **Statistical rigor** — bootstrap validation to separate real disparities from sampling noise, rather than reporting point estimates as conclusions.
- **Explainable AI** — per-decision SHAP attributions alongside model-level coefficients.
- **Written governance communication** — translating notebook findings into a report legible to a non-technical reviewer (`reports/governance_report.md`).

It is intended as a portfolio piece for roles in AI governance and compliance, ML fairness/audit engineering, and applied data science in regulated industries.

## Publish guidance

This repository is designed as a reproducible fairness investigation, not a production model. If you publish it, frame the narrative around audit findings, uncertainty, and data limitations.
