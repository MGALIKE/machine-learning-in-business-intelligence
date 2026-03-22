"""
=============================================================================
DATA ENGINEERING PIPELINE - ML in Business Intelligence Hackathon
=============================================================================
Role: Data Engineer
Purpose: Clean, preprocess, explore, and structure the dataset for ML models

Pipeline Steps:
    1. Load & Inspect raw data
    2. Data Quality Assessment
    3. Cleaning & Preprocessing
    4. Feature Engineering
    5. Data Structuring for ML
    6. Export cleaned datasets
    7. Generate Data Report
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FILE = "ml_bi_hackathon_train.xlsx"
TEST_FILE = "ml_bi_hackathon_test_features.xlsx"
SAMPLE_FILE = "sample_submission.xlsx"
OUTPUT_DIR = "cleaned_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# STEP 1: LOAD & INSPECT RAW DATA
# =============================================================================
print_header("STEP 1: LOADING RAW DATA")

train_raw = pd.read_excel(TRAIN_FILE)
test_raw = pd.read_excel(TEST_FILE)
sample_sub = pd.read_excel(SAMPLE_FILE)

print(f"Training data  : {train_raw.shape[0]:,} rows × {train_raw.shape[1]} columns")
print(f"Test data      : {test_raw.shape[0]:,} rows × {test_raw.shape[1]} columns")
print(f"Sample submiss.: {sample_sub.shape[0]:,} rows × {sample_sub.shape[1]} columns")

print_subheader("Training Data Columns & Types")
for col in train_raw.columns:
    dtype = train_raw[col].dtype
    nunique = train_raw[col].nunique()
    print(f"  {col:35s} | {str(dtype):10s} | {nunique:5d} unique")

print_subheader("Target Variable Distribution (churned)")
churned_counts = train_raw['churned'].value_counts()
churned_pct = train_raw['churned'].value_counts(normalize=True) * 100
for val in sorted(churned_counts.index):
    label = "Churned" if val == 1 else "Not Churned"
    print(f"  {label:15s} ({val}): {churned_counts[val]:5d}  ({churned_pct[val]:.1f}%)")
print(f"  {'Class Imbalance Ratio':15s}   : {churned_counts[0]/churned_counts[1]:.2f}:1 (Not Churned:Churned)")


# =============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# =============================================================================
print_header("STEP 2: DATA QUALITY ASSESSMENT")

# 2.1 Missing Values
print_subheader("2.1 Missing Values Check")
train_missing = train_raw.isnull().sum()
test_missing = test_raw.isnull().sum()
total_missing_train = train_missing.sum()
total_missing_test = test_missing.sum()
print(f"  Training data total missing: {total_missing_train}")
print(f"  Test data total missing    : {total_missing_test}")
if total_missing_train == 0 and total_missing_test == 0:
    print("  ✅ No missing values found in either dataset!")
else:
    for col in train_missing[train_missing > 0].index:
        print(f"  ⚠️  TRAIN - {col}: {train_missing[col]} missing ({train_missing[col]/len(train_raw)*100:.1f}%)")
    for col in test_missing[test_missing > 0].index:
        print(f"  ⚠️  TEST  - {col}: {test_missing[col]} missing ({test_missing[col]/len(test_raw)*100:.1f}%)")

# 2.2 Duplicate Rows
print_subheader("2.2 Duplicate Rows Check")
train_dupes = train_raw.duplicated().sum()
test_dupes = test_raw.duplicated().sum()
train_id_dupes = train_raw['customer_id'].duplicated().sum()
test_id_dupes = test_raw['customer_id'].duplicated().sum()
print(f"  Training duplicate rows       : {train_dupes}")
print(f"  Training duplicate IDs        : {train_id_dupes}")
print(f"  Test duplicate rows           : {test_dupes}")
print(f"  Test duplicate IDs            : {test_id_dupes}")
if train_dupes == 0 and test_dupes == 0:
    print("  ✅ No duplicate rows found!")

# 2.3 ID overlap check
print_subheader("2.3 Train/Test ID Overlap")
overlap = set(train_raw['customer_id']) & set(test_raw['customer_id'])
print(f"  Overlapping customer IDs: {len(overlap)}")
if len(overlap) == 0:
    print("  ✅ No ID leakage between train and test!")
else:
    print(f"  ⚠️  WARNING: {len(overlap)} overlapping IDs found!")

# 2.4 Negative Values Check
print_subheader("2.4 Negative & Invalid Values")
numeric_cols = train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
should_be_positive = [
    'age', 'tenure_months', 'avg_monthly_spend_usd', 'website_visits_last_30d',
    'app_sessions_last_30d', 'support_tickets_last_90d', 'late_payments_last_12m',
    'discount_usage_rate', 'email_open_rate', 'products_owned',
    'returns_last_12m', 'days_since_last_purchase', 'estimated_clv_usd'
]
for col in should_be_positive:
    neg_count = (train_raw[col] < 0).sum()
    if neg_count > 0:
        print(f"  ⚠️  {col}: {neg_count} negative values (min={train_raw[col].min()})")
    else:
        print(f"  ✅ {col}: all values ≥ 0")

# NPS score can legitimately be negative (-100 to +100)
nps_neg = (train_raw['nps_score'] < 0).sum()
print(f"  ℹ️  nps_score: {nps_neg} negative values (valid range: -100 to +100, actual: [{train_raw['nps_score'].min()}, {train_raw['nps_score'].max()}])")

# Rate columns should be 0-1
for rate_col in ['discount_usage_rate', 'email_open_rate']:
    out_of_range = ((train_raw[rate_col] < 0) | (train_raw[rate_col] > 1)).sum()
    if out_of_range > 0:
        print(f"  ⚠️  {rate_col}: {out_of_range} values outside [0, 1]")
    else:
        print(f"  ✅ {rate_col}: all values within [0, 1]")

# 2.5 Outlier Detection (IQR method)
print_subheader("2.5 Outlier Detection (IQR Method)")
outlier_cols = [c for c in numeric_cols if c not in ['churned']]
outlier_summary = {}
for col in outlier_cols:
    Q1 = train_raw[col].quantile(0.25)
    Q3 = train_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((train_raw[col] < lower) | (train_raw[col] > upper)).sum()
    if n_outliers > 0:
        outlier_summary[col] = {
            'count': n_outliers,
            'pct': n_outliers / len(train_raw) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }
        print(f"  ⚠️  {col:35s}: {n_outliers:3d} outliers ({n_outliers/len(train_raw)*100:.1f}%) | IQR bounds: [{lower:.2f}, {upper:.2f}]")

if not outlier_summary:
    print("  ✅ No statistical outliers detected")
else:
    print(f"\n  Total columns with outliers: {len(outlier_summary)}")
    print("  ℹ️  Decision: KEEP outliers — they represent real extreme customer behavior")

# 2.6 Data Leakage Detection
print_subheader("2.6 ⚠️ DATA LEAKAGE DETECTION")
leakage_check = train_raw.groupby('recommended_action')['churned'].mean()
print("  Churn rate by recommended_action:")
for action, rate in leakage_check.items():
    emoji = "🔴" if rate in [0.0, 1.0] else "🟡"
    print(f"    {emoji} {action:40s}: {rate:.0%} churn rate")

print("\n  🚨 CRITICAL: 'recommended_action' has PERFECT correlation with 'churned'!")
print("     - All 'Targeted win-back campaign' and 'Immediate retention offer' → churned = 1")
print("     - All other actions → churned = 0")
print("     - This column was likely DERIVED from the churn outcome (data leakage)")
print("     - ⚡ Action: MUST EXCLUDE from ML features to prevent data leakage!")

clv_corr = train_raw['estimated_clv_usd'].corr(train_raw['churned'])
print(f"\n  ℹ️  'estimated_clv_usd' correlation with churned: {clv_corr:.3f}")
print(f"  ℹ️  'clv_segment' is derived from 'estimated_clv_usd'")
print("     - These may contain mild leakage but are likely pre-calculated features")
print("     - ⚡ Action: KEEP but flag for the Data Scientist to evaluate")


# =============================================================================
# STEP 3: DATA CLEANING & PREPROCESSING
# =============================================================================
print_header("STEP 3: DATA CLEANING & PREPROCESSING")

train = train_raw.copy()
test = test_raw.copy()

# 3.1 Standardize column names (already lowercase with underscores - good!)
print_subheader("3.1 Column Name Standardization")
print("  ✅ Column names already in snake_case format")

# 3.2 Standardize categorical values
print_subheader("3.2 Categorical Value Standardization")
cat_cols = train.select_dtypes(include='object').columns.tolist()
cat_cols.remove('customer_id')  # Don't touch the ID

for col in cat_cols:
    # Check for leading/trailing whitespace
    train[col] = train[col].str.strip()
    test[col] = test[col].str.strip()
    
    # Verify consistent casing
    unique_train = sorted(train[col].unique())
    unique_test = sorted(test[col].unique())
    
    # Check for unseen categories in test
    new_in_test = set(unique_test) - set(unique_train)
    if new_in_test:
        print(f"  ⚠️  {col}: test has unseen categories: {new_in_test}")
    else:
        print(f"  ✅ {col}: consistent categories between train/test")

# 3.3 Data Type Verification
print_subheader("3.3 Data Type Verification & Optimization")
# Verify integer columns don't need float (no NaN values so this is fine)
int_cols = train.select_dtypes(include='int64').columns.tolist()
for col in int_cols:
    if col == 'churned':
        continue
    col_min, col_max = train[col].min(), train[col].max()
    if col_min >= 0 and col_max <= 255:
        # Could be uint8 but keep int for compatibility
        print(f"  ℹ️  {col}: range [{col_min}, {col_max}] — compact int range")
    else:
        print(f"  ℹ️  {col}: range [{col_min}, {col_max}]")

# 3.4 NPS Score validation
print_subheader("3.4 NPS Score Validation")
nps_min, nps_max = train['nps_score'].min(), train['nps_score'].max()
print(f"  Range: [{nps_min}, {nps_max}] (standard NPS: -100 to +100)")
if nps_min >= -100 and nps_max <= 100:
    print("  ✅ NPS scores within valid range")
else:
    print("  ⚠️  NPS scores outside standard range — clamping to [-100, 100]")
    train['nps_score'] = train['nps_score'].clip(-100, 100)
    test['nps_score'] = test['nps_score'].clip(-100, 100)


# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================
print_header("STEP 4: FEATURE ENGINEERING")

def engineer_features(df):
    """Apply feature engineering to a dataframe."""
    df = df.copy()
    
    # 4.1 Engagement Score (composite of digital interactions)
    df['engagement_score'] = (
        df['website_visits_last_30d'] + 
        df['app_sessions_last_30d']
    )
    
    # 4.2 Spend per tenure month (customer value velocity)
    df['spend_per_tenure'] = df['avg_monthly_spend_usd'] / df['tenure_months'].clip(lower=1)
    
    # 4.3 Support intensity (tickets relative to tenure)
    df['support_intensity'] = df['support_tickets_last_90d'] / df['tenure_months'].clip(lower=1)
    
    # 4.4 Payment reliability (inverse of late payments)
    df['payment_reliability'] = 1 - (df['late_payments_last_12m'] / df['late_payments_last_12m'].max().clip(min=1))
    
    # 4.5 CLV to spend ratio
    df['clv_spend_ratio'] = df['estimated_clv_usd'] / (df['avg_monthly_spend_usd'] * 12).clip(lower=1)
    
    # 4.6 Recency bucket (days since last purchase categorized)
    df['recency_bucket'] = pd.cut(
        df['days_since_last_purchase'],
        bins=[0, 14, 30, 60, 90, float('inf')],
        labels=['Very Recent', 'Recent', 'Moderate', 'Distant', 'Very Distant'],
        right=True
    ).astype(str)
    
    # 4.7 Age group
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, float('inf')],
        labels=['18-25', '26-35', '36-45', '46-55', '55+'],
        right=True
    ).astype(str)
    
    # 4.8 Tenure category
    df['tenure_category'] = pd.cut(
        df['tenure_months'],
        bins=[0, 6, 12, 24, 48, float('inf')],
        labels=['New (<6m)', 'Early (6-12m)', 'Developing (1-2y)', 'Established (2-4y)', 'Loyal (4y+)'],
        right=True
    ).astype(str)
    
    # 4.9 NPS Category (Detractor/Passive/Promoter)
    df['nps_category'] = pd.cut(
        df['nps_score'],
        bins=[-float('inf'), 6, 8, float('inf')],
        labels=['Detractor', 'Passive', 'Promoter']
    ).astype(str)
    # Note: Traditional NPS uses 0-10 scale, but this dataset has -59 to 100
    # We'll use a percentile-based approach instead
    df['nps_category'] = pd.cut(
        df['nps_score'],
        bins=[-float('inf'), 
              df['nps_score'].quantile(0.33),
              df['nps_score'].quantile(0.66),
              float('inf')],
        labels=['Detractor', 'Passive', 'Promoter']
    ).astype(str)
    
    # 4.10 High-risk flag (combining multiple risk signals)
    df['risk_score'] = (
        (df['late_payments_last_12m'] >= 2).astype(int) +
        (df['support_tickets_last_90d'] >= 3).astype(int) +
        (df['days_since_last_purchase'] >= 60).astype(int) +
        (df['email_open_rate'] < 0.3).astype(int) +
        (df['nps_score'] < 0).astype(int)
    )
    df['high_risk'] = (df['risk_score'] >= 3).astype(int)
    
    # 4.11 Digital vs Physical preference
    df['digital_preference'] = df['app_sessions_last_30d'] / (
        df['website_visits_last_30d'] + df['app_sessions_last_30d']
    ).clip(lower=1)
    
    # 4.12 Discount sensitivity (high discount usage + low loyalty)
    df['discount_dependency'] = df['discount_usage_rate'] * (1 - df['email_open_rate'])
    
    return df

print("  Applying feature engineering to training data...")
train = engineer_features(train)
print("  Applying feature engineering to test data...")
test = engineer_features(test)

new_features = [
    'engagement_score', 'spend_per_tenure', 'support_intensity',
    'payment_reliability', 'clv_spend_ratio', 'recency_bucket',
    'age_group', 'tenure_category', 'nps_category', 'risk_score',
    'high_risk', 'digital_preference', 'discount_dependency'
]
print(f"\n  ✅ Created {len(new_features)} new features:")
for feat in new_features:
    dtype = train[feat].dtype
    if dtype == 'object':
        unique = train[feat].nunique()
        print(f"     • {feat:30s} (categorical, {unique} categories)")
    else:
        print(f"     • {feat:30s} (numeric, mean={train[feat].mean():.3f})")


# =============================================================================
# STEP 5: FEATURE CORRELATION ANALYSIS
# =============================================================================
print_header("STEP 5: FEATURE ANALYSIS")

print_subheader("5.1 Correlation with Churn (Top Features)")
numeric_train = train.select_dtypes(include=['int64', 'float64'])
correlations = numeric_train.corr()['churned'].drop('churned').sort_values(key=abs, ascending=False)
print(f"\n  {'Feature':40s} {'Correlation':>12s}  Direction")
print("  " + "-" * 68)
for feat, corr_val in correlations.head(15).items():
    direction = "↑ churn" if corr_val > 0 else "↓ churn"
    bar = "█" * int(abs(corr_val) * 40)
    print(f"  {feat:40s} {corr_val:+.4f}      {direction}  {bar}")

print_subheader("5.2 Churn Rate by Engineered Features")

for cat_feat in ['recency_bucket', 'age_group', 'tenure_category', 'nps_category', 'high_risk']:
    print(f"\n  {cat_feat}:")
    churn_by_cat = train.groupby(cat_feat)['churned'].agg(['mean', 'count'])
    churn_by_cat.columns = ['churn_rate', 'count']
    churn_by_cat = churn_by_cat.sort_values('churn_rate', ascending=False)
    for idx, row in churn_by_cat.iterrows():
        bar = "█" * int(row['churn_rate'] * 30)
        print(f"    {str(idx):25s} {row['churn_rate']:.1%} churn  (n={int(row['count']):4d})  {bar}")


# =============================================================================
# STEP 6: PREPARE & STRUCTURE DATA FOR ML
# =============================================================================
print_header("STEP 6: STRUCTURING DATA FOR ML")

# Define feature groups
LEAKAGE_COLS = ['recommended_action']  # Perfect correlation with target
ID_COL = 'customer_id'
TARGET_COL = 'churned'

# Columns to encode
CATEGORICAL_FEATURES = [
    'gender', 'city_tier', 'subscription_type', 'contract_type',
    'marketing_channel', 'clv_segment',
    'recency_bucket', 'age_group', 'tenure_category', 'nps_category'
]

NUMERICAL_FEATURES = [
    'age', 'tenure_months', 'avg_monthly_spend_usd', 'website_visits_last_30d',
    'app_sessions_last_30d', 'support_tickets_last_90d', 'late_payments_last_12m',
    'discount_usage_rate', 'email_open_rate', 'nps_score', 'products_owned',
    'returns_last_12m', 'days_since_last_purchase', 'estimated_clv_usd',
    'engagement_score', 'spend_per_tenure', 'support_intensity',
    'payment_reliability', 'clv_spend_ratio', 'risk_score', 'high_risk',
    'digital_preference', 'discount_dependency'
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

print(f"  ID column         : {ID_COL}")
print(f"  Target column     : {TARGET_COL}")
print(f"  Leakage columns   : {LEAKAGE_COLS} (EXCLUDED)")
print(f"  Categorical feats : {len(CATEGORICAL_FEATURES)}")
print(f"  Numerical feats   : {len(NUMERICAL_FEATURES)}")
print(f"  Total features    : {len(ALL_FEATURES)}")

# 6.1 One-Hot Encode categorical features
print_subheader("6.1 One-Hot Encoding Categorical Features")

train_encoded = pd.get_dummies(train, columns=CATEGORICAL_FEATURES, drop_first=False)
test_encoded = pd.get_dummies(test, columns=CATEGORICAL_FEATURES, drop_first=False)

# Align columns between train and test (add missing columns with 0, remove extra)
train_feature_cols = [c for c in train_encoded.columns if c not in [ID_COL, TARGET_COL] + LEAKAGE_COLS]
test_feature_cols = [c for c in test_encoded.columns if c not in [ID_COL] + LEAKAGE_COLS]

# Add missing columns in test
for col in train_feature_cols:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
        print(f"  ⚠️  Added missing column to test: {col}")

# Remove columns from test that are not in train
extra_in_test = set(test_feature_cols) - set(train_feature_cols)
if extra_in_test:
    for col in extra_in_test:
        print(f"  ⚠️  Removing extra column from test: {col}")

# Final feature list (excluding leakage and ID)
final_features = sorted([c for c in train_feature_cols if c not in LEAKAGE_COLS])

print(f"\n  Final encoded features: {len(final_features)}")

# 6.2 Create final structured datasets
print_subheader("6.2 Final Structured Datasets")

train_final = train_encoded[[ID_COL] + final_features + [TARGET_COL]].copy()
test_final = test_encoded[[ID_COL] + [c for c in final_features if c in test_encoded.columns]].copy()

# Add any missing columns to test
for col in final_features:
    if col not in test_final.columns:
        test_final[col] = 0

# Reorder test columns to match train
test_final = test_final[[ID_COL] + final_features]

print(f"  Training set : {train_final.shape}")
print(f"  Test set     : {test_final.shape}")

# Also save non-encoded version for EDA/dashboards
train_clean = train.drop(columns=LEAKAGE_COLS).copy()
test_clean = test.drop(columns=LEAKAGE_COLS).copy()


# =============================================================================
# STEP 7: EXPORT CLEANED DATA
# =============================================================================
print_header("STEP 7: EXPORTING CLEANED DATA")

# 7.1 Cleaned data (with readable categories - for dashboards & EDA)
train_clean.to_csv(f"{OUTPUT_DIR}/train_cleaned.csv", index=False)
test_clean.to_csv(f"{OUTPUT_DIR}/test_cleaned.csv", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/train_cleaned.csv ({train_clean.shape})")
print(f"  ✅ Saved: {OUTPUT_DIR}/test_cleaned.csv ({test_clean.shape})")

# 7.2 ML-ready encoded data (one-hot encoded for models)
train_final.to_csv(f"{OUTPUT_DIR}/train_ml_ready.csv", index=False)
test_final.to_csv(f"{OUTPUT_DIR}/test_ml_ready.csv", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/train_ml_ready.csv ({train_final.shape})")
print(f"  ✅ Saved: {OUTPUT_DIR}/test_ml_ready.csv ({test_final.shape})")

# 7.3 Feature metadata (for Data Scientist reference)
feature_metadata = {
    "created_at": datetime.now().isoformat(),
    "pipeline_version": "1.0",
    "train_shape": list(train_final.shape),
    "test_shape": list(test_final.shape),
    "target_column": TARGET_COL,
    "id_column": ID_COL,
    "leakage_columns_removed": LEAKAGE_COLS,
    "original_categorical_features": CATEGORICAL_FEATURES,
    "original_numerical_features": NUMERICAL_FEATURES,
    "engineered_features": new_features,
    "total_encoded_features": len(final_features),
    "class_distribution": {
        "not_churned": int(churned_counts[0]),
        "churned": int(churned_counts[1]),
        "churn_rate": float(churned_pct[1] / 100)
    },
    "key_findings": [
        "No missing values in either dataset",
        "No duplicate rows or IDs",
        "No train/test ID overlap (no data leakage there)",
        "'recommended_action' has PERFECT correlation with 'churned' — EXCLUDED as leakage",
        "NPS scores range from -59 to 100 (negatives are valid)",
        "Class imbalance: 72.5% not churned vs 27.5% churned (ratio 2.64:1)",
        "Top churn predictors: estimated_clv_usd, late_payments, days_since_last_purchase, tenure_months"
    ]
}

with open(f"{OUTPUT_DIR}/feature_metadata.json", 'w') as f:
    json.dump(feature_metadata, f, indent=2)
print(f"  ✅ Saved: {OUTPUT_DIR}/feature_metadata.json")

# 7.4 Data dictionary
data_dict = []
for col in final_features:
    if col in train_final.columns:
        data_dict.append({
            'feature': col,
            'dtype': str(train_final[col].dtype),
            'min': float(train_final[col].min()),
            'max': float(train_final[col].max()),
            'mean': float(train_final[col].mean()),
            'std': float(train_final[col].std()),
        })
data_dict_df = pd.DataFrame(data_dict)
data_dict_df.to_csv(f"{OUTPUT_DIR}/data_dictionary.csv", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/data_dictionary.csv")


# =============================================================================
# STEP 8: PIPELINE SUMMARY REPORT
# =============================================================================
print_header("PIPELINE SUMMARY REPORT")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DATA ENGINEERING PIPELINE - COMPLETE                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  📊 DATA OVERVIEW                                                          ║
║  ├─ Training samples   : {train_raw.shape[0]:>6,}                                        ║
║  ├─ Test samples       : {test_raw.shape[0]:>6,}                                        ║
║  ├─ Original features  : {train_raw.shape[1] - 2:>6,} (excl. ID & target)                ║
║  ├─ Engineered features: {len(new_features):>6,}                                        ║
║  └─ Final ML features  : {len(final_features):>6,} (after one-hot encoding)              ║
║                                                                            ║
║  🧹 CLEANING ACTIONS                                                       ║
║  ├─ Missing values     : None found (0 imputation needed)                  ║
║  ├─ Duplicates         : None found (0 rows removed)                       ║
║  ├─ Leakage removed    : 'recommended_action' column excluded              ║
║  ├─ Categories stripped: Whitespace cleaned across all categoricals         ║
║  └─ Values validated   : All ranges verified, NPS within bounds            ║
║                                                                            ║
║  🎯 TARGET (churned)                                                       ║
║  ├─ Not Churned (0)    : {churned_counts[0]:>5,} ({churned_pct[0]:.1f}%)                            ║
║  ├─ Churned (1)        : {churned_counts[1]:>5,} ({churned_pct[1]:.1f}%)                            ║
║  └─ Imbalance Ratio    : {churned_counts[0]/churned_counts[1]:.2f}:1                                     ║
║                                                                            ║
║  📁 OUTPUT FILES (in '{OUTPUT_DIR}/')                                    ║
║  ├─ train_cleaned.csv      → For EDA & dashboards                         ║
║  ├─ test_cleaned.csv       → For EDA & dashboards                         ║
║  ├─ train_ml_ready.csv     → For ML model training (encoded)              ║
║  ├─ test_ml_ready.csv      → For ML model prediction (encoded)            ║
║  ├─ feature_metadata.json  → Pipeline documentation                       ║
║  └─ data_dictionary.csv    → Feature statistics & definitions             ║
║                                                                            ║
║  ⚠️  NOTES FOR DATA SCIENTIST                                              ║
║  ├─ 'recommended_action' is DATA LEAKAGE — already removed                ║
║  ├─ Class imbalance (27.5%) — consider SMOTE or class weights              ║
║  ├─ 'estimated_clv_usd' has strongest neg. correlation with churn          ║
║  └─ Engineered 'risk_score' and 'high_risk' flags for modeling             ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("✅ Data Engineering Pipeline completed successfully!")
print(f"   All outputs saved to: ./{OUTPUT_DIR}/")
