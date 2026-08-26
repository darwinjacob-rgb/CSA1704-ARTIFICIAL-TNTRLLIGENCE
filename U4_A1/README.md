# Diabetes Diagnosis & RL-Based Treatment Recommendation

Healthcare AI/ML case study: predictive diagnosis of diabetes (Decision Tree +
Logistic Regression) and a Q-Learning reinforcement-learning agent for
personalised treatment recommendation (Diet / Exercise / Medication / Monitor).

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `1_Problem_Statement.pdf` | All four tasks (data prep, decision tree, statistical learning, reinforcement learning) as originally specified. |
| 2 | `2_Solution.pdf` | Full task-by-task written solution with explanations, justifications, and actual computed results/tables. |
| 3 | `diabetes_ml_project.py` | Complete, runnable Python implementation of the whole pipeline. |
| 4 | `diabetes_ml_output.png` | Consolidated visual output: class balance, decision tree, confusion matrix, ROC curves, feature importance, Q-table heatmap. |
| 5 | `5_Project_Report.pdf` | Consolidated professional project report (executive summary, methodology, results, conclusion) combining all tasks. |
| 6 | `README.md` | This file. |

## How to run the code

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
python diabetes_ml_project.py
```

This regenerates `diabetes_ml_output.png` and `results.json` (the single
source of truth used to build the PDF reports, so all reported numbers stay
consistent with the code).

## Pipeline overview

**Task 1 — Data Preparation & Inductive Learning**
- Synthetic, clinically-plausible dataset (1,200 patients: Age, BMI, Glucose,
  Cholesterol, Blood Pressure, Family History → Diabetic).
- 70:30 stratified train/test split.
- Class imbalance corrected with **SMOTE** (training set only).
- Continuous features standardised (z-score); Family History already binary.

**Task 2 — Decision Tree for Diagnosis**
- CART Decision Tree (Gini index), root split on Glucose.
- Metrics: Accuracy, Precision, Recall, F1-Score, confusion matrix, False
  Negative analysis with clinical mitigation recommendations.
- Cost-complexity **post-pruning** with pre- vs. post-pruning comparison.

**Task 3 — Statistical Learning for Risk Stratification**
- Logistic Regression, compared against the Decision Tree on Accuracy and
  AUC-ROC.
- Feature-importance comparison (Gini importance vs. absolute coefficients),
  top-3 predictors identified for both models.

**Task 4 — Reinforcement Learning for Treatment Recommendation**
- MDP formulation: States = {Critical, HighRisk, ModerateRisk, Healthy},
  Actions = {Diet, Exercise, Medication, Monitor}, reward = health-improvement
  score, stochastic action-dependent transitions.
- Tabular **Q-Learning** (α=0.1, γ=0.9, ε-greedy) trained across many patient
  episodes with a convergence-tolerance stopping criterion.
- Final learned policy extracted per health stage, with a discussion of
  patient-safety, bias, and explainability considerations for clinical
  deployment.

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn, imbalanced-learn

## Notes

- The dataset is **synthetically generated** (see `generate_synthetic_dataset`
  in `diabetes_ml_project.py`) to keep this case study self-contained and
  reproducible; swap in a real (IRB-approved) clinical dataset with the same
  column names to run the identical pipeline on real data.
- All numbers quoted in the Solution and Report PDFs are produced directly by
  the script — nothing is hand-typed — so re-running the script and
  regenerating the PDFs will always stay in sync.
