# German Credit Bias Audit

This repository contains a reproducible fairness audit of a German credit dataset.
The core analysis is implemented in `bias_audit.ipynb` and compares model behavior when protected attributes are hidden versus visible.

## Contents

- `bias_audit.ipynb` — main notebook with data loading, model training, fairness metrics, and audit narrative.
- `german_credit_raw.csv` — raw dataset used for the audit.
- `bias_audit_results.xlsx` — saved audit results from the notebook.
- `bias_audit_results_both.xlsx` — comparison results for both hidden and visible protected-attribute runs.
- `bias_audit_stability_summary.xlsx` — stability and bootstrap analysis for random-state variation and noise estimation.

## What this project shows

- Logistic regression is used to classify credit risk.
- Protected attributes include `sex`, `age`, and `foreign_worker`.
- The notebook compares two settings:
  - `DROP_PROTECTED=True` — protected attributes removed from model inputs.
  - `DROP_PROTECTED=False` — protected attributes included in the model.
- The audit measures selection-rate gaps, disparate impact, and stability across random seeds.
- Bootstrap analysis estimates whether observed gaps are stable or likely due to sampling noise.

## Findings

- Overall accuracy is stable in both settings (~0.75).
- The visible-protected model shows a larger sex gap and lower disparate impact ratio.
- The hidden-protected model shows a smaller sex gap, but proxy features still carry some signal.
- Age gaps are less stable and should be reported as uncertain.
- The notebook emphasizes that data limitations and label bias remain critical issues.

## How to run

1. Install dependencies:
   ```bash
   python3 -m pip install pandas scikit-learn openpyxl
   ```
2. Open `bias_audit.ipynb` in Jupyter or a compatible notebook viewer.
3. Run the notebook cells in order.
4. Use the `DROP_PROTECTED` switch to compare audit modes.

## Notes and limitations

- The dataset is from 1994 Germany, so it reflects historical credit decisions.
- Some subgroups are small, making fairness ratios noisy.
- Sex categories are binary and incomplete.
- The labels are bank decisions, which may themselves encode unfairness.

## Publish guidance

This repository is designed as a reproducible fairness investigation, not a production model. If you publish it, frame the narrative around audit findings, uncertainty, and data limitations.
