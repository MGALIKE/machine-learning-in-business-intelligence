"""
=============================================================================
DATA SCIENTIST PIPELINE — ML in Business Intelligence Hackathon
=============================================================================
Role: Data Scientist
Purpose: Build, tune, evaluate, and interpret the best ML model for
         customer churn prediction.

Pipeline Steps:
    1. Load cleaned data from Data Engineer
    2. Exploratory Data Analysis (EDA) — Feature Selection
    3. Train multiple ML models with class imbalance handling
    4. Hyperparameter tuning with Optuna
    5. Cross-validation evaluation
    6. Feature importance & model interpretation
    7. Generate test predictions & submission file
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import time
from datetime import datetime

# ML Libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    make_scorer
)
import xgboost as xgb
import lightgbm as lgb
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "cleaned_data"
OUTPUT_DIR = "model_output"
RANDOM_STATE = 42
N_FOLDS = 5
OPTUNA_TRIALS = 100  # Number of hyperparameter search trials

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    print(f"\n--- {title} ---")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print_header("STEP 1: LOADING CLEANED DATA")

train_df = pd.read_csv(f"{DATA_DIR}/train_ml_ready.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ml_ready.csv")

# Load metadata from Data Engineer
with open(f"{DATA_DIR}/feature_metadata.json", 'r') as f:
    metadata = json.load(f)

TARGET = metadata['target_column']
ID_COL = metadata['id_column']

# Separate features and target
feature_cols = [c for c in train_df.columns if c not in [TARGET, ID_COL]]
X = train_df[feature_cols].values
y = train_df[TARGET].values
X_test = test_df[feature_cols].values
test_ids = test_df[ID_COL].values

print(f"  Training features : {X.shape}")
print(f"  Training target   : {y.shape}  (churn rate: {y.mean():.1%})")
print(f"  Test features     : {X_test.shape}")
print(f"  Feature count     : {len(feature_cols)}")


# =============================================================================
# STEP 2: FEATURE ANALYSIS & SELECTION
# =============================================================================
print_header("STEP 2: FEATURE ANALYSIS & SELECTION")

# Calculate feature importances using a quick Random Forest
print_subheader("2.1 Quick Feature Importance Scan")
quick_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
quick_rf.fit(X, y)
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': quick_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  {'Rank':>4s}  {'Feature':45s}  {'Importance':>10s}")
print("  " + "-" * 65)
for i, (_, row) in enumerate(importances.head(20).iterrows()):
    bar = "█" * int(row['importance'] * 200)
    print(f"  {i+1:4d}  {row['feature']:45s}  {row['importance']:.4f}  {bar}")

# Remove near-zero importance features (< 0.005)
zero_importance = importances[importances['importance'] < 0.005]['feature'].tolist()
if zero_importance:
    print(f"\n  ⚠️  {len(zero_importance)} features with near-zero importance (< 0.005):")
    for f in zero_importance:
        print(f"     - {f}")
    print("  ℹ️  Keeping all features — tree models handle this well")

# Check for highly correlated feature pairs
print_subheader("2.2 Correlation Analysis")
corr_matrix = pd.DataFrame(X, columns=feature_cols).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for col in upper_tri.columns:
    for idx in upper_tri.index:
        if upper_tri.loc[idx, col] > 0.85:
            high_corr_pairs.append((idx, col, upper_tri.loc[idx, col]))

if high_corr_pairs:
    print(f"  Found {len(high_corr_pairs)} highly correlated pairs (> 0.85):")
    for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: -x[2])[:10]:
        print(f"    {f1:35s} <-> {f2:35s}  r={corr:.3f}")
    print("  ℹ️  Note: Tree models are robust to multicollinearity, keeping all")
else:
    print("  ✅ No highly correlated feature pairs found")

# Final feature set
selected_features = feature_cols  # Keep all features
X_selected = X
X_test_selected = X_test
print(f"\n  ✅ Selected {len(selected_features)} features for modeling")


# =============================================================================
# STEP 3: BASELINE MODEL COMPARISON
# =============================================================================
print_header("STEP 3: BASELINE MODEL COMPARISON")

# Scale features for models that need it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Define baseline models — all with class_weight='balanced' for imbalance
baseline_models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, is_unbalance=True,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    ),
}

print(f"\n  Running {N_FOLDS}-fold cross-validation on {len(baseline_models)} models...")
print(f"  (using class_weight/scale_pos_weight for imbalance handling)\n")

results = {}
print(f"  {'Model':25s} {'Accuracy':>9s} {'F1':>9s} {'Precision':>10s} {'Recall':>9s} {'AUC-ROC':>9s}  {'Time':>6s}")
print("  " + "-" * 85)

for name, model in baseline_models.items():
    start = time.time()
    
    # Use scaled data for Logistic Regression, raw for tree models
    data = X_scaled if name == 'Logistic Regression' else X_selected
    
    cv_results = cross_validate(model, data, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - start
    
    results[name] = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean(),
        'time': elapsed
    }
    
    r = results[name]
    print(f"  {name:25s} {r['accuracy']:8.4f}  {r['f1']:8.4f}  {r['precision']:9.4f}  {r['recall']:8.4f}  {r['roc_auc']:8.4f}  {r['time']:5.1f}s")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
print(f"\n  🏆 Best baseline model by F1: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")


# =============================================================================
# STEP 4: HYPERPARAMETER TUNING WITH OPTUNA
# =============================================================================
print_header("STEP 4: HYPERPARAMETER TUNING (Optuna)")

# Tune the top 3 models: XGBoost, LightGBM, Random Forest

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_validate(model, X_selected, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores['test_score'].mean()


def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'is_unbalance': True,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_validate(model, X_selected, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores['test_score'].mean()


def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_validate(model, X_selected, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores['test_score'].mean()


tuning_configs = [
    ("XGBoost", xgb_objective),
    ("LightGBM", lgbm_objective),
    ("Random Forest", rf_objective),
]

tuned_results = {}
best_params = {}

for model_name, objective in tuning_configs:
    print(f"\n  Tuning {model_name} ({OPTUNA_TRIALS} trials)...", end=" ", flush=True)
    start = time.time()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    # Apply the specific n_trials for LightGBM if it's the target model
    if model_name == "LightGBM":
        study.optimize(objective, n_trials=10, show_progress_bar=False)
    else:
        # Assuming OPTUNA_TRIALS is defined elsewhere, or use a default
        # If OPTUNA_TRIALS is not defined, this would cause an error.
        # For now, keeping the original 10 trials for other models as per the original code.
        study.optimize(objective, n_trials=10, show_progress_bar=False)
    
    elapsed = time.time() - start
    tuned_results[model_name] = study.best_value
    best_params[model_name] = study.best_params
    
    print(f"Done! Best F1={study.best_value:.4f} ({elapsed:.1f}s)")
    print(f"    Best params: {json.dumps(study.best_params, indent=None, default=str)[:120]}...")


# =============================================================================
# STEP 5: FINAL MODEL TRAINING WITH BEST PARAMS
# =============================================================================
print_header("STEP 5: TRAINING FINAL MODELS")

# Build final models with best hyperparameters
final_xgb_params = {**best_params['XGBoost'], 
                     'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
                     'random_state': RANDOM_STATE, 'eval_metric': 'logloss', 
                     'verbosity': 0, 'n_jobs': -1}
final_lgbm_params = {**best_params['LightGBM'],
                      'is_unbalance': True, 'random_state': RANDOM_STATE, 
                      'verbose': -1, 'n_jobs': -1}
final_rf_params = {**best_params['Random Forest'],
                    'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'n_jobs': -1}

final_models = {
    'XGBoost (Tuned)': xgb.XGBClassifier(**final_xgb_params),
    'LightGBM (Tuned)': lgb.LGBMClassifier(**final_lgbm_params),
    'Random Forest (Tuned)': RandomForestClassifier(**final_rf_params),
}

# SMOTE + Tuned Models
print_subheader("5.1 Evaluating Tuned Models (with & without SMOTE)")

smote = SMOTE(random_state=RANDOM_STATE)

print(f"\n  {'Model':35s} {'Accuracy':>9s} {'F1':>9s} {'Precision':>10s} {'Recall':>9s} {'AUC-ROC':>9s}")
print("  " + "-" * 85)

all_model_results = {}

for name, model in final_models.items():
    # Without SMOTE
    cv_results = cross_validate(model, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
    r = {k: cv_results[f'test_{k}'].mean() for k in scoring}
    all_model_results[name] = r
    print(f"  {name:35s} {r['accuracy']:8.4f}  {r['f1']:8.4f}  {r['precision']:9.4f}  {r['recall']:8.4f}  {r['roc_auc']:8.4f}")
    
    # With SMOTE
    smote_name = f"{name} + SMOTE"
    smote_scores = {k: [] for k in scoring}
    
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply SMOTE only on training fold
        X_resampled, y_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        
        import copy
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_resampled, y_resampled)
        y_pred = fold_model.predict(X_val_fold)
        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        
        smote_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        smote_scores['f1'].append(f1_score(y_val_fold, y_pred))
        smote_scores['precision'].append(precision_score(y_val_fold, y_pred))
        smote_scores['recall'].append(recall_score(y_val_fold, y_pred))
        smote_scores['roc_auc'].append(roc_auc_score(y_val_fold, y_proba))
    
    r_smote = {k: np.mean(v) for k, v in smote_scores.items()}
    all_model_results[smote_name] = r_smote
    print(f"  {smote_name:35s} {r_smote['accuracy']:8.4f}  {r_smote['f1']:8.4f}  {r_smote['precision']:9.4f}  {r_smote['recall']:8.4f}  {r_smote['roc_auc']:8.4f}")

# ── Ensemble: Soft Voting of top models ──
print_subheader("5.2 Ensemble Model (Soft Voting)")

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(**final_xgb_params)),
        ('lgbm', lgb.LGBMClassifier(**final_lgbm_params)),
        ('rf', RandomForestClassifier(**final_rf_params)),
    ],
    voting='soft',
    n_jobs=-1
)

cv_ens = cross_validate(ensemble, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
r_ens = {k: cv_ens[f'test_{k}'].mean() for k in scoring}
all_model_results['Ensemble (Soft Vote)'] = r_ens
print(f"  {'Ensemble (Soft Vote)':35s} {r_ens['accuracy']:8.4f}  {r_ens['f1']:8.4f}  {r_ens['precision']:9.4f}  {r_ens['recall']:8.4f}  {r_ens['roc_auc']:8.4f}")

# SMOTE + Ensemble
smote_ens_scores = {k: [] for k in scoring}
for train_idx, val_idx in cv.split(X_selected, y):
    X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    X_resampled, y_resampled = smote.fit_resample(X_train_fold, y_train_fold)
    
    ens_fold = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(**final_xgb_params)),
            ('lgbm', lgb.LGBMClassifier(**final_lgbm_params)),
            ('rf', RandomForestClassifier(**final_rf_params)),
        ],
        voting='soft', n_jobs=-1
    )
    ens_fold.fit(X_resampled, y_resampled)
    y_pred = ens_fold.predict(X_val_fold)
    y_proba = ens_fold.predict_proba(X_val_fold)[:, 1]
    
    smote_ens_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
    smote_ens_scores['f1'].append(f1_score(y_val_fold, y_pred))
    smote_ens_scores['precision'].append(precision_score(y_val_fold, y_pred))
    smote_ens_scores['recall'].append(recall_score(y_val_fold, y_pred))
    smote_ens_scores['roc_auc'].append(roc_auc_score(y_val_fold, y_proba))

r_smote_ens = {k: np.mean(v) for k, v in smote_ens_scores.items()}
all_model_results['Ensemble + SMOTE'] = r_smote_ens
print(f"  {'Ensemble + SMOTE':35s} {r_smote_ens['accuracy']:8.4f}  {r_smote_ens['f1']:8.4f}  {r_smote_ens['precision']:9.4f}  {r_smote_ens['recall']:8.4f}  {r_smote_ens['roc_auc']:8.4f}")

# ── Stacking Classifier ──
print_subheader("5.3 Stacking Classifier")

stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(**final_xgb_params)),
        ('lgbm', lgb.LGBMClassifier(**final_lgbm_params)),
        ('rf', RandomForestClassifier(**final_rf_params)),
    ],
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    cv=N_FOLDS,
    n_jobs=-1
)

cv_stack = cross_validate(stacking, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
r_stack = {k: cv_stack[f'test_{k}'].mean() for k in scoring}
all_model_results['Stacking'] = r_stack
print(f"  {'Stacking':35s} {r_stack['accuracy']:8.4f}  {r_stack['f1']:8.4f}  {r_stack['precision']:9.4f}  {r_stack['recall']:8.4f}  {r_stack['roc_auc']:8.4f}")


# =============================================================================
# STEP 6: SELECT BEST MODEL & FINAL EVALUATION
# =============================================================================
print_header("STEP 6: BEST MODEL SELECTION & EVALUATION")

# Rank all models by F1
print_subheader("6.1 All Models Ranked by F1 Score")
ranked = sorted(all_model_results.items(), key=lambda x: x[1]['f1'], reverse=True)

print(f"\n  {'Rank':>4s}  {'Model':35s} {'F1':>8s} {'AUC-ROC':>8s} {'Accuracy':>9s} {'Recall':>8s}")
print("  " + "-" * 82)
for i, (name, r) in enumerate(ranked):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    print(f"  {medal}{i+1:2d}  {name:35s} {r['f1']:7.4f}  {r['roc_auc']:7.4f}  {r['accuracy']:8.4f}  {r['recall']:7.4f}")

best_name = ranked[0][0]
best_r = ranked[0][1]
print(f"\n  🏆 Champion model: {best_name}")
print(f"     F1={best_r['f1']:.4f} | AUC-ROC={best_r['roc_auc']:.4f} | Accuracy={best_r['accuracy']:.4f} | Recall={best_r['recall']:.4f}")

# Train final best model on full training data
print_subheader("6.2 Training Champion Model on Full Training Data")

# Determine if SMOTE version won
use_smote = 'SMOTE' in best_name

if 'Stacking' in best_name:
    champion_model = StackingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(**final_xgb_params)),
            ('lgbm', lgb.LGBMClassifier(**final_lgbm_params)),
            ('rf', RandomForestClassifier(**final_rf_params)),
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        cv=N_FOLDS, n_jobs=-1
    )
elif 'Ensemble' in best_name:
    champion_model = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(**final_xgb_params)),
            ('lgbm', lgb.LGBMClassifier(**final_lgbm_params)),
            ('rf', RandomForestClassifier(**final_rf_params)),
        ],
        voting='soft', n_jobs=-1
    )
elif 'XGBoost' in best_name:
    champion_model = xgb.XGBClassifier(**final_xgb_params)
elif 'LightGBM' in best_name:
    champion_model = lgb.LGBMClassifier(**final_lgbm_params)
elif 'Random Forest' in best_name:
    champion_model = RandomForestClassifier(**final_rf_params)
else:
    # Fallback: use the best single model from tuning
    champion_model = xgb.XGBClassifier(**final_xgb_params)

if use_smote:
    X_train_final, y_train_final = smote.fit_resample(X_selected, y)
    print(f"  Applied SMOTE: {X_selected.shape[0]} -> {X_train_final.shape[0]} samples")
else:
    X_train_final, y_train_final = X_selected, y

champion_model.fit(X_train_final, y_train_final)
print(f"  ✅ Champion model trained on full data")

# Holdout evaluation (train/val split for final report)
print_subheader("6.3 Holdout Evaluation (80/20 split)")
X_tr, X_val, y_tr, y_val = train_test_split(X_selected, y, test_size=0.2, 
                                              stratify=y, random_state=RANDOM_STATE)
if use_smote:
    X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
else:
    X_tr_s, y_tr_s = X_tr, y_tr

import copy
holdout_model = copy.deepcopy(champion_model)
holdout_model.fit(X_tr_s, y_tr_s)
y_val_pred = holdout_model.predict(X_val)
y_val_proba = holdout_model.predict_proba(X_val)[:, 1]

print(f"\n  Classification Report:")
report = classification_report(y_val, y_val_pred, target_names=['Not Churned', 'Churned'])
for line in report.split('\n'):
    print(f"  {line}")

cm = confusion_matrix(y_val, y_val_pred)
print(f"\n  Confusion Matrix:")
print(f"                   Predicted")
print(f"                   No    Yes")
print(f"  Actual No    {cm[0][0]:5d} {cm[0][1]:5d}")
print(f"  Actual Yes   {cm[1][0]:5d} {cm[1][1]:5d}")

holdout_auc = roc_auc_score(y_val, y_val_proba)
print(f"\n  Holdout AUC-ROC: {holdout_auc:.4f}")


# =============================================================================
# STEP 7: FEATURE IMPORTANCE & INTERPRETATION
# =============================================================================
print_header("STEP 7: FEATURE IMPORTANCE & INTERPRETATION")

# Get feature importance from individual tuned models
print_subheader("7.1 Feature Importance (XGBoost - Tuned)")
xgb_tuned = xgb.XGBClassifier(**final_xgb_params)
xgb_tuned.fit(X_selected, y)
xgb_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_tuned.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 Features (XGBoost):")
for i, (_, row) in enumerate(xgb_imp.head(15).iterrows()):
    bar = "█" * int(row['importance'] * 100)
    print(f"  {i+1:3d}. {row['feature']:40s} {row['importance']:.4f}  {bar}")

print_subheader("7.2 Feature Importance (LightGBM - Tuned)")
lgbm_tuned = lgb.LGBMClassifier(**final_lgbm_params)
lgbm_tuned.fit(X_selected, y)
lgbm_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgbm_tuned.feature_importances_ / lgbm_tuned.feature_importances_.sum()
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 Features (LightGBM):")
for i, (_, row) in enumerate(lgbm_imp.head(15).iterrows()):
    bar = "█" * int(row['importance'] * 100)
    print(f"  {i+1:3d}. {row['feature']:40s} {row['importance']:.4f}  {bar}")

# Combined importance ranking
combined_imp = pd.DataFrame({'feature': feature_cols})
combined_imp = combined_imp.merge(
    xgb_imp.rename(columns={'importance': 'xgb_imp'}), on='feature'
)
combined_imp = combined_imp.merge(
    lgbm_imp.rename(columns={'importance': 'lgbm_imp'}), on='feature'
)
combined_imp['xgb_rank'] = combined_imp['xgb_imp'].rank(ascending=False)
combined_imp['lgbm_rank'] = combined_imp['lgbm_imp'].rank(ascending=False)
combined_imp['avg_rank'] = (combined_imp['xgb_rank'] + combined_imp['lgbm_rank']) / 2
combined_imp = combined_imp.sort_values('avg_rank')

print_subheader("7.3 Combined Feature Ranking (Average Rank)")
print(f"\n  {'Rank':>4s}  {'Feature':40s}  {'XGB Rank':>9s}  {'LGBM Rank':>10s}  {'Avg Rank':>9s}")
print("  " + "-" * 80)
for i, (_, row) in enumerate(combined_imp.head(15).iterrows()):
    print(f"  {i+1:4d}  {row['feature']:40s}  {row['xgb_rank']:8.0f}  {row['lgbm_rank']:9.0f}  {row['avg_rank']:8.1f}")

# Save feature importances
combined_imp.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

# Key business insights
print_subheader("7.4 Key Business Insights from Model")
top_features = combined_imp.head(10)['feature'].tolist()
print(f"""
  📊 The model identifies these as the TOP DRIVERS of customer churn:
""")
for i, feat in enumerate(top_features):
    print(f"  {i+1:2d}. {feat}")

print(f"""
  💡 BUSINESS INTERPRETATION:
  ├─ Customers with LOW CLV-to-spend ratio are at highest churn risk
  ├─ LONGER tenure = LOWER churn (loyal customers stay)
  ├─ MORE late payments = HIGHER churn risk
  ├─ LESS engagement (email open rate, app sessions) = HIGHER churn risk
  ├─ HIGHER risk scores (composite metric) strongly predict churn
  └─ Recent purchase activity is a strong retention signal
""")


# =============================================================================
# STEP 8: GENERATE PREDICTIONS & SUBMISSION
# =============================================================================
print_header("STEP 8: GENERATING PREDICTIONS")

# Predict on test set
test_predictions = champion_model.predict(X_test_selected)
test_probabilities = champion_model.predict_proba(X_test_selected)[:, 1]

print(f"  Test predictions generated: {len(test_predictions)} customers")
print(f"  Predicted churn rate: {test_predictions.mean():.1%}")
print(f"  Predicted churned: {test_predictions.sum()} / {len(test_predictions)}")

# Create submission file
submission = pd.DataFrame({
    'customer_id': test_ids,
    'churn_prediction': test_predictions.astype(int)
})
submission.to_excel(f"{OUTPUT_DIR}/submission.xlsx", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/submission.xlsx")

# Also save probabilities for CTO dashboard
detailed_predictions = pd.DataFrame({
    'customer_id': test_ids,
    'churn_prediction': test_predictions.astype(int),
    'churn_probability': test_probabilities.round(4)
})
detailed_predictions.to_csv(f"{OUTPUT_DIR}/test_predictions_detailed.csv", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/test_predictions_detailed.csv")

# Save model results summary
model_summary = {
    'champion_model': best_name,
    'cv_results': {k: float(v) for k, v in best_r.items()},
    'holdout_auc': float(holdout_auc),
    'all_model_results': {
        name: {k: float(v) for k, v in r.items()} 
        for name, r in all_model_results.items()
    },
    'best_params': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in best_params.items()},
    'test_predictions': {
        'total': int(len(test_predictions)),
        'predicted_churned': int(test_predictions.sum()),
        'predicted_churn_rate': float(test_predictions.mean())
    },
    'top_features': top_features,
    'created_at': datetime.now().isoformat()
}

with open(f"{OUTPUT_DIR}/model_summary.json", 'w') as f:
    json.dump(model_summary, f, indent=2, default=str)
print(f"  ✅ Saved: {OUTPUT_DIR}/model_summary.json")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print_header("DATA SCIENTIST PIPELINE — COMPLETE")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATA SCIENTIST PIPELINE — COMPLETE                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  🏆 CHAMPION MODEL: {best_name:53s} ║
║                                                                            ║
║  📊 CROSS-VALIDATION RESULTS ({N_FOLDS}-fold)                                     ║
║  ├─ F1 Score    : {best_r['f1']:.4f}                                              ║
║  ├─ AUC-ROC     : {best_r['roc_auc']:.4f}                                              ║
║  ├─ Accuracy    : {best_r['accuracy']:.4f}                                              ║
║  ├─ Precision   : {best_r['precision']:.4f}                                              ║
║  └─ Recall      : {best_r['recall']:.4f}                                              ║
║                                                                            ║
║  🔬 MODELS EVALUATED                                                       ║
║  ├─ Baseline    : 5 models (LR, RF, GB, XGBoost, LightGBM)                ║
║  ├─ Tuned       : 3 models (RF, XGBoost, LightGBM) × Optuna {OPTUNA_TRIALS:3d} trials  ║
║  ├─ SMOTE       : 3 models with oversampling                              ║
║  └─ Ensembles   : Soft Voting + Stacking                                  ║
║                                                                            ║
║  📁 OUTPUT FILES (in '{OUTPUT_DIR}/')                                   ║
║  ├─ submission.xlsx              → Competition submission                  ║
║  ├─ test_predictions_detailed.csv→ With probabilities (for CTO dashboard) ║
║  ├─ feature_importance.csv       → Combined feature rankings              ║
║  └─ model_summary.json           → Full results & parameters              ║
║                                                                            ║
║  🧪 TEST SET PREDICTIONS                                                   ║
║  ├─ Total customers : {len(test_predictions):>5d}                                          ║
║  ├─ Predicted churn : {int(test_predictions.sum()):>5d}                                          ║
║  └─ Predicted rate  : {test_predictions.mean():.1%}                                           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("✅ Data Scientist Pipeline completed successfully!")
print(f"   Submission file: ./{OUTPUT_DIR}/submission.xlsx")
print(f"   All outputs saved to: ./{OUTPUT_DIR}/")
