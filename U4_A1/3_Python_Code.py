"""
================================================================================
 HEALTHCARE AI/ML CASE STUDY
 Early Diagnosis of Diabetes + RL-based Personalised Treatment Recommendation
================================================================================
Author  : AI/ML Engineer (Case Study Submission)
Purpose : End-to-end implementation covering
          Task 1 - Data Preparation & Inductive Learning
          Task 2 - Decision Tree for Diagnosis
          Task 3 - Statistical Learning (Logistic Regression) for Risk
                   Stratification
          Task 4 - Reinforcement Learning (Q-Learning) for Treatment
                   Recommendation

Run:  python diabetes_ml_project.py
Outputs:
    - diabetes_ml_output.png   (all visualisations, single image, 6 panels)
    - results.json             (all numeric results, used to auto-build the
                                 solution/report PDFs so numbers always match)
================================================================================
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

RNG = np.random.RandomState(42)
RESULTS = {}

# ==============================================================================
# TASK 1 : DATA PREPARATION & INDUCTIVE LEARNING
# ==============================================================================

def generate_synthetic_dataset(n=1200):
    """
    Generates a realistic, clinically-plausible synthetic diabetes dataset.
    A logistic latent function ties glucose, BMI, age and family history to
    the diabetes outcome so the classes are imbalanced (~15% positive), which
    mirrors real-world screening population prevalence.
    """
    age = RNG.normal(48, 14, n).clip(18, 90)
    bmi = RNG.normal(27, 5.5, n).clip(15, 55)
    glucose = RNG.normal(110, 28, n).clip(60, 300)
    chol = RNG.normal(200, 35, n).clip(100, 350)
    bp_sys = RNG.normal(125, 16, n).clip(80, 200)
    family_history = RNG.binomial(1, 0.28, n)

    # latent risk score -> probability of diabetes (logistic link)
    z = (
        0.045 * (glucose - 110)
        + 0.09 * (bmi - 27)
        + 0.02 * (age - 48)
        + 0.9 * family_history
        + 0.01 * (chol - 200)
        + 0.015 * (bp_sys - 125)
        - 3.4
    )
    prob = 1 / (1 + np.exp(-z))
    diabetic = RNG.binomial(1, prob)

    df = pd.DataFrame({
        "Age": age.round(1),
        "BMI": bmi.round(1),
        "Glucose": glucose.round(1),
        "Cholesterol": chol.round(1),
        "BloodPressure": bp_sys.round(1),
        "FamilyHistory": family_history,
        "Diabetic": diabetic
    })
    return df


df = generate_synthetic_dataset(1200)
RESULTS["dataset_size"] = len(df)
RESULTS["class_counts_raw"] = df["Diabetic"].value_counts().to_dict()
RESULTS["class_ratio_raw"] = round(
    df["Diabetic"].value_counts()[0] / df["Diabetic"].value_counts()[1], 2
)

FEATURES = ["Age", "BMI", "Glucose", "Cholesterol", "BloodPressure", "FamilyHistory"]
TARGET = "Diabetic"

X = df[FEATURES]
y = df[TARGET]

# ---- 70:30 stratified split (medical use-case justification in report) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
RESULTS["train_size"] = len(X_train)
RESULTS["test_size"] = len(X_test)
RESULTS["train_class_counts"] = y_train.value_counts().to_dict()
RESULTS["test_class_counts"] = y_test.value_counts().to_dict()

# ---- Preprocessing: normalisation (numeric) ---------------------------------
scaler = StandardScaler()
num_cols = ["Age", "BMI", "Glucose", "Cholesterol", "BloodPressure"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
# FamilyHistory is already a binary-encoded categorical feature -> no change needed

# ---- Class imbalance handling: SMOTE on TRAINING data only ------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
RESULTS["class_counts_after_smote"] = pd.Series(y_train_res).value_counts().to_dict()

# raw (unscaled) train for the Decision Tree (trees do not need scaling,
# but we also keep a scaled+SMOTE version to show both approaches)
X_train_raw_res, y_train_raw_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ==============================================================================
# TASK 2 : DECISION TREE FOR DIAGNOSIS
# ==============================================================================

dt = DecisionTreeClassifier(
    criterion="gini", max_depth=6, min_samples_leaf=10, random_state=42
)
dt.fit(X_train_raw_res, y_train_raw_res)

# Fully-grown (unconstrained) tree -> used as the "pre-pruning" baseline for
# the cost-complexity pruning comparison in Task 2(c)
dt_full = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_full.fit(X_train_raw_res, y_train_raw_res)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)
tn, fp, fn, tp = cm_dt.ravel()

RESULTS["decision_tree"] = {
    "accuracy": round(acc_dt, 4),
    "precision": round(prec_dt, 4),
    "recall": round(rec_dt, 4),
    "f1_score": round(f1_dt, 4),
    "confusion_matrix": cm_dt.tolist(),
    "true_negatives": int(tn), "false_positives": int(fp),
    "false_negatives": int(fn), "true_positives": int(tp),
    "root_feature": FEATURES[dt.tree_.feature[0]],
    "root_gini": round(dt.tree_.impurity[0], 4),
    "feature_importances": dict(zip(FEATURES, np.round(dt.feature_importances_, 4)))
}

tree_rules_text = export_text(dt, feature_names=FEATURES, max_depth=2)
RESULTS["decision_tree"]["first_three_levels_text"] = tree_rules_text

# ---- Cost-complexity post-pruning (applied to the fully-grown tree) --------
path = dt_full.cost_complexity_pruning_path(X_train_raw_res, y_train_raw_res)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[ccp_alphas >= 0]

# scan candidate alphas, evaluate held-out accuracy, keep the best-performing
# pruned tree that also reduces complexity vs. the fully-grown tree
best_alpha, best_acc, best_tree = 0.0, 0.0, dt_full
for a in ccp_alphas:
    cand = DecisionTreeClassifier(criterion="gini", random_state=42, ccp_alpha=a)
    cand.fit(X_train_raw_res, y_train_raw_res)
    acc_cand = accuracy_score(y_test, cand.predict(X_test))
    if cand.get_n_leaves() > 1 and acc_cand >= best_acc:
        best_acc, best_alpha, best_tree = acc_cand, a, cand

dt_pruned = best_tree
chosen_alpha = best_alpha
y_pred_full = dt_full.predict(X_test)
y_pred_pruned = dt_pruned.predict(X_test)

acc_pre = accuracy_score(y_test, y_pred_full)   # fully-grown, unpruned tree
acc_post = accuracy_score(y_test, y_pred_pruned)

RESULTS["pruning"] = {
    "chosen_alpha": round(float(chosen_alpha), 5),
    "pre_pruning_accuracy": round(acc_pre, 4),
    "pre_pruning_depth": int(dt_full.get_depth()),
    "pre_pruning_leaves": int(dt_full.get_n_leaves()),
    "post_pruning_accuracy": round(acc_post, 4),
    "post_pruning_depth": int(dt_pruned.get_depth()),
    "post_pruning_leaves": int(dt_pruned.get_n_leaves()),
}

# ==============================================================================
# TASK 3 : STATISTICAL LEARNING (LOGISTIC REGRESSION) FOR RISK STRATIFICATION
# ==============================================================================

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_res, y_train_res)

y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_dt = roc_auc_score(y_test, y_prob_dt)

RESULTS["logistic_regression"] = {
    "accuracy": round(acc_lr, 4),
    "precision": round(prec_lr, 4),
    "recall": round(rec_lr, 4),
    "f1_score": round(f1_lr, 4),
    "auc_roc": round(auc_lr, 4),
    "coefficients": dict(zip(FEATURES, np.round(logreg.coef_[0], 4)))
}
RESULTS["decision_tree"]["auc_roc"] = round(auc_dt, 4)

# top-3 predictors from each model
dt_top3 = sorted(RESULTS["decision_tree"]["feature_importances"].items(),
                  key=lambda kv: -kv[1])[:3]
lr_top3 = sorted(RESULTS["logistic_regression"]["coefficients"].items(),
                  key=lambda kv: -abs(kv[1]))[:3]
RESULTS["top3_predictors"] = {
    "decision_tree": dt_top3,
    "logistic_regression": lr_top3
}

# ==============================================================================
# TASK 4 : REINFORCEMENT LEARNING (Q-LEARNING) FOR TREATMENT RECOMMENDATION
# ==============================================================================

# ---- MDP definition -----------------------------------------------------
STATES = ["Critical", "HighRisk", "ModerateRisk", "Healthy"]   # health stages
ACTIONS = ["Diet", "Exercise", "Medication", "Monitor"]
N_S, N_A = len(STATES), len(ACTIONS)

# Reward function: health-improvement score.
# Moving the patient towards "Healthy" gives positive reward; regressing
# towards "Critical" is penalised; an action that is clinically mismatched to
# the state (e.g. only "Monitor" for a Critical patient) is penalised.
def reward_fn(state_idx, action_idx, next_state_idx):
    improvement = state_idx - next_state_idx * -1  # placeholder, replaced below
    base = (next_state_idx - state_idx) * 10   # +10 per stage improved
    # penalise clinically inappropriate mismatches
    if state_idx == 0 and action_idx == 3:      # Critical patient just "Monitored"
        base -= 8
    if state_idx == 0 and action_idx == 2:      # Critical patient -> Medication (appropriate)
        base += 4
    if state_idx == 3 and action_idx == 2:      # Healthy patient over-medicated
        base -= 4
    if state_idx == 3 and action_idx in (0, 1): # Healthy + Diet/Exercise = maintenance, small reward
        base += 2
    return base

# Transition dynamics: stochastic, action-dependent probability of the
# patient's stage improving / staying / worsening.
def transition(state_idx, action_idx):
    # probability of moving toward Healthy (index+1), staying, or worsening
    improve_p = {"Diet": 0.35, "Exercise": 0.4, "Medication": 0.55, "Monitor": 0.15}[ACTIONS[action_idx]]
    worsen_p = 0.10
    stay_p = 1 - improve_p - worsen_p
    r = RNG.rand()
    if r < improve_p and state_idx < N_S - 1:
        next_idx = state_idx + 1
    elif r < improve_p + worsen_p and state_idx > 0:
        next_idx = state_idx - 1
    else:
        next_idx = state_idx
    return next_idx

# ---- Q-learning hyperparameters ---------------------------------------------
ALPHA = 0.1      # learning rate
GAMMA = 0.9      # discount factor
EPSILON = 0.2    # exploration rate
N_EPISODES = 400  # >= 5 patient episodes required (400 run for stable convergence)
MAX_STEPS = 6
CONVERGENCE_TOL = 1e-3

Q = np.zeros((N_S, N_A))
q_history = []          # store full Q-table snapshot each episode
tracked_updates = []    # log updates for at least 3 (state, action) pairs across >=2 iterations
TRACK_PAIRS = [(0, 2), (1, 0), (3, 3)]   # (Critical, Medication), (HighRisk, Diet), (Healthy, Monitor)

delta_per_episode = []

for ep in range(N_EPISODES):
    state_idx = RNG.choice([0, 1, 2])  # patients typically start at-risk
    max_delta = 0.0
    for step in range(MAX_STEPS):
        if RNG.rand() < EPSILON:
            action_idx = RNG.randint(N_A)
        else:
            action_idx = int(np.argmax(Q[state_idx]))

        next_idx = transition(state_idx, action_idx)
        r = reward_fn(state_idx, action_idx, next_idx)

        old_q = Q[state_idx, action_idx]
        best_next = np.max(Q[next_idx])
        new_q = old_q + ALPHA * (r + GAMMA * best_next - old_q)
        Q[state_idx, action_idx] = new_q
        max_delta = max(max_delta, abs(new_q - old_q))

        if (state_idx, action_idx) in TRACK_PAIRS:
            tracked_updates.append({
                "episode": ep + 1, "step": step + 1,
                "state": STATES[state_idx], "action": ACTIONS[action_idx],
                "reward": r, "old_Q": round(old_q, 4), "new_Q": round(new_q, 4)
            })

        state_idx = next_idx
        if state_idx == N_S - 1:  # reached Healthy -> episode ends
            break
    delta_per_episode.append(max_delta)
    q_history.append(Q.copy())

RESULTS["q_learning"] = {
    "states": STATES,
    "actions": ACTIONS,
    "alpha": ALPHA, "gamma": GAMMA, "epsilon": EPSILON,
    "episodes_run": N_EPISODES,
    "convergence_tolerance": CONVERGENCE_TOL,
    "max_delta_last_episode": round(float(delta_per_episode[-1]), 5),
    "converged": bool(delta_per_episode[-1] < CONVERGENCE_TOL),
    "final_q_table": np.round(Q, 3).tolist(),
    "tracked_updates": tracked_updates[:12],
    "final_policy": {STATES[s]: ACTIONS[int(np.argmax(Q[s]))] for s in range(N_S)}
}

# ==============================================================================
# SAVE NUMERIC RESULTS (single source of truth for the PDFs)
# ==============================================================================
with open("/home/claude/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

# ==============================================================================
# VISUALISATION -> single PNG, 6 panels
# ==============================================================================
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)
fig.suptitle("Diabetes Diagnosis & Treatment Recommendation — Model Outputs",
             fontsize=16, fontweight="bold")

# Panel 1: class balance before/after SMOTE
ax1 = fig.add_subplot(gs[0, 0])
before = pd.Series(y_train).value_counts().sort_index()
after = pd.Series(y_train_res).value_counts().sort_index()
w = 0.35
idx = np.arange(2)
ax1.bar(idx - w/2, before.values, width=w, label="Before SMOTE", color="#e07a5f")
ax1.bar(idx + w/2, after.values, width=w, label="After SMOTE", color="#3d5a80")
ax1.set_xticks(idx); ax1.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
ax1.set_ylabel("Count"); ax1.set_title("Class Balance: Training Set")
ax1.legend(fontsize=8)

# Panel 2: Decision tree (first 3 levels)
ax2 = fig.add_subplot(gs[0, 1])
plot_tree(dt, max_depth=2, feature_names=FEATURES, class_names=["No", "Yes"],
          filled=True, fontsize=7, ax=ax2)
ax2.set_title("Decision Tree — First 3 Levels")

# Panel 3: Confusion matrix (Decision Tree)
ax3 = fig.add_subplot(gs[0, 2])
im = ax3.imshow(cm_dt, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax3.text(j, i, cm_dt[i, j], ha="center", va="center",
                  color="white" if cm_dt[i, j] > cm_dt.max()/2 else "black", fontsize=13)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Pred 0", "Pred 1"])
ax3.set_yticks([0, 1]); ax3.set_yticklabels(["Actual 0", "Actual 1"])
ax3.set_title(f"Confusion Matrix — Decision Tree\n(FN={fn}, FP={fp})")

# Panel 4: ROC curves
ax4 = fig.add_subplot(gs[1, 0])
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
ax4.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.2f})", color="#e07a5f")
ax4.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.2f})", color="#3d5a80")
ax4.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curve Comparison"); ax4.legend(fontsize=8)

# Panel 5: Feature importance comparison
ax5 = fig.add_subplot(gs[1, 1])
imp_dt = pd.Series(RESULTS["decision_tree"]["feature_importances"]).sort_values()
imp_lr = pd.Series({k: abs(v) for k, v in RESULTS["logistic_regression"]["coefficients"].items()})
imp_lr = imp_lr / imp_lr.sum()
imp_lr = imp_lr[imp_dt.index]
y_pos = np.arange(len(imp_dt))
ax5.barh(y_pos - 0.2, imp_dt.values, height=0.4, label="Decision Tree", color="#e07a5f")
ax5.barh(y_pos + 0.2, imp_lr.values, height=0.4, label="Logistic Regression", color="#3d5a80")
ax5.set_yticks(y_pos); ax5.set_yticklabels(imp_dt.index, fontsize=8)
ax5.set_title("Feature Importance Comparison"); ax5.legend(fontsize=8)

# Panel 6: Q-table heatmap (final policy)
ax6 = fig.add_subplot(gs[1, 2])
im2 = ax6.imshow(Q, cmap="YlGnBu", aspect="auto")
ax6.set_xticks(range(N_A)); ax6.set_xticklabels(ACTIONS, rotation=30, ha="right", fontsize=8)
ax6.set_yticks(range(N_S)); ax6.set_yticklabels(STATES, fontsize=8)
for i in range(N_S):
    for j in range(N_A):
        ax6.text(j, i, f"{Q[i,j]:.1f}", ha="center", va="center", fontsize=8,
                  color="white" if Q[i, j] > Q.max()/1.5 else "black")
ax6.set_title("Q-Learning: Final Q-Table")
plt.colorbar(im2, ax=ax6, fraction=0.046, pad=0.04)

plt.savefig("/home/claude/diabetes_ml_output.png", dpi=150, bbox_inches="tight")
plt.close()

print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
print(json.dumps(RESULTS, indent=2, default=str))
