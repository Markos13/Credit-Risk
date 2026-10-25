import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path


df = pd.read_csv(
    "loan_2014_18.csv",
    low_memory=False
)


df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
)


df["issue_year"] = df["issue_d"].dt.year

#print("Missing issuance dates:", df["issue_d"].isna().sum())

valid_status = [
    "Fully Paid",
    "Charged Off",
    "Default"
]

df_clean = df[
    df["loan_status"].isin(valid_status)
].copy()

df_clean["default"] = df_clean["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
})



leakage_cols = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "loan_status",
    "default",
    "id",
    "url",
    "Unnamed: 0",
    "pymnt_plan",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "funded_amnt_inv"
]



# Calculate yearly loan statistics
yearly_summary = df_clean.groupby("issue_year")["default"].agg(
    total_loans="count",
    total_defaults="sum",
    default_rate="mean"
)

yearly_summary["default_rate"] *= 100


#print("\nYearly default statistics:")
#print(yearly_summary)



plt.figure(figsize=(10, 5))

plt.plot(
    yearly_summary.index,
    yearly_summary["default_rate"],
    marker="o"
)

plt.xlabel("Loan Issuance Year")
plt.ylabel("Observed Default Rate (%)")
plt.title("Observed Default Rate by Loan Issuance Year")
plt.grid(alpha=0.3)


plt.xticks(yearly_summary.index.astype(int))
plt.tight_layout()

Path("reports/figures").mkdir(parents=True, exist_ok=True)

plt.savefig(
    "reports/figures/default_rate_by_year.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Each year's loans-repayment time(36 or 60 months) in percentage 
term_distribution = pd.crosstab(
    df_clean["issue_year"],
    df_clean["term"],
    normalize="index"
) * 100


#print("\nLoan term distribution:")
#print(term_distribution.round(2))


status_by_year = pd.crosstab(
    df["issue_year"],
    df["loan_status"],
    normalize="index"
) * 100


'''print("\nLoan status by year:")
print(status_by_year.round(2).to_string())

print("\nRemaining feature count:", df_model.shape[1])

print("\nRemaining columns:")
for col in df_model.columns:
    print(col)'''


additional_exclusions = [
    # Underwriting and funding information
    "funded_amnt",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "initial_list_status",
    "verification_status",
    "verification_status_joint",

    # Dates used for chronological splitting
    "issue_d",
    "issue_year",

    # High-cardinality or unnecessary variables
    "emp_title",
    "title",
    "zip_code",
    "policy_code",
    "term",
    
]

df_model = df_clean.drop(
    columns=leakage_cols + additional_exclusions,
    errors="ignore"
).copy()


#print("Remaining features:", df_model.shape[1])


'''df_clean["earliest_cr_line"] = pd.to_datetime(
    df_clean["earliest_cr_line"],
    format="%m-%d-%y",
    errors="coerce"
)

invalid_credit_dates = (
    df_clean["earliest_cr_line"] >
    df_clean["issue_d"]
).sum()

print(
    "Credit history dates after loan issuance:",
    invalid_credit_dates
)'''



# ========================================
# 9. Select eligible loans
# ========================================

eligible_mask = (
    (df_clean["term"].str.strip() == "36 months") &
    (df_clean["issue_year"].between(2014, 2016))
)

df_selected = df_clean.loc[eligible_mask].copy()

# Model features and target
X = df_model.loc[df_selected.index].copy()
y = df_selected["default"].copy()


# ========================================
# 10. Chronological split
# ========================================

train_mask = df_selected["issue_year"] == 2014
val_mask = df_selected["issue_year"] == 2015
test_mask = df_selected["issue_year"] == 2016

X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_val = y.loc[val_mask].copy()
y_test = y.loc[test_mask].copy()


# ========================================
# 11. Verify the split
# ========================================

'''print("\nDataset shapes:")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nDefault rates:")

print("Training:", round(y_train.mean() * 100, 2), "%")
print("Validation:", round(y_val.mean() * 100, 2), "%")
print("Testing:", round(y_test.mean() * 100, 2), "%")

print("\nMissing target values:", y.isna().sum())

print("\nOverlapping indexes:")
print("Train / Validation:", len(X_train.index.intersection(X_val.index)))
print("Train / Test:", len(X_train.index.intersection(X_test.index)))
print("Validation / Test:", len(X_val.index.intersection(X_test.index)))'''


# ========================================
# 12. Missing-value analysis
# ========================================

# Calculate missing percentages using TRAINING data only
missing_pct = X_train.isna().mean() * 100

# Identify features with more than 30% missing values
excluded_features = missing_pct[
    missing_pct > 30
].sort_values(ascending=False)

# Identify features to keep
features_to_keep = missing_pct[
    missing_pct <= 30
].index

# Apply the SAME feature selection to all datasets
X_train = X_train[features_to_keep].copy()
X_val = X_val[features_to_keep].copy()
X_test = X_test[features_to_keep].copy()



# ========================================
# 14. Feature transformations
# ========================================

for dataset in [X_train, X_val, X_test]:

    # Convert earliest credit line date
    credit_date = pd.to_datetime(
        dataset["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce"
    )

    # Get each loan's issuance date
    issue_date = df_clean.loc[dataset.index, "issue_d"]

    # Calculate credit history length in months
    dataset["credit_history_months"] = (
        (issue_date.dt.year - credit_date.dt.year) * 12
        + (issue_date.dt.month - credit_date.dt.month)
    )


    # Remove original date column
    del dataset["earliest_cr_line"]


    # Convert revolving utilization to numeric
    dataset["revol_util"] = pd.to_numeric(
        dataset["revol_util"]
        .astype("string")
        .str.replace("%", "", regex=False),
        errors="coerce"
    )


# ========================================
# 15. Verify feature selection
# ========================================

'''print("\nExcluded features (>30% missing in training):")
print(excluded_features.to_string())

print("\nNumber of excluded features:", len(excluded_features))

print("\nDataset shapes after filtering:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ========================================
# 16. Inspect remaining feature types
# ========================================

print("\nRemaining feature data types:")
print(X_train.dtypes.to_string())

print("\nRemaining missing values (%):")
print(
    (X_train.isna().mean() * 100)
    .sort_values(ascending=False)
    .round(2)
    .to_string()
)

print("\nCategorical and date columns:")
print(
    X_train.select_dtypes(
        include=["object", "str", "category", "datetime"]
    ).columns.tolist()
)'''


# ========================================
# 17. Pearson correlation analysis
# ========================================

# Select numerical features from TRAINING data only
numeric_features = X_train.select_dtypes(
    include="number"
)

# Calculate correlations between features and default
target_corr = numeric_features.corrwith(y_train)

# Sort by absolute correlation strength
target_corr = target_corr.reindex(
    target_corr.abs().sort_values(ascending=False).index
)

#print("\nPearson correlation with default:")
#print(target_corr.round(4).to_string())


# ========================================
# 18. Highly correlated feature pairs
# ========================================

import numpy as np

# Calculate feature correlation matrix
corr_matrix = numeric_features.corr().abs()

# Keep only upper triangle to avoid duplicates
upper_triangle = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

# Find feature pairs with correlation > 0.90
high_corr_pairs = (
    upper_triangle.stack()
    .loc[lambda x: x > 0.90]
    .sort_values(ascending=False)
)

#print("\nHighly correlated feature pairs:")
#print(high_corr_pairs.to_string())


# ========================================
# 18. Remove redundant features
# ========================================

redundant_features = [
    "fico_range_high",
    "num_sats",
    "num_rev_tl_bal_gt_0"
]

X_train = X_train.drop(columns=redundant_features)
X_val = X_val.drop(columns=redundant_features)
X_test = X_test.drop(columns=redundant_features)

'''print("\nShapes after removing redundant features:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)'''


# ========================================
# 19. Identify feature types
# ========================================

numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

'''print("\nNumerical features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))

print("\nCategorical columns:")
print(categorical_cols)'''

assert len(numeric_cols) + len(categorical_cols) == X_train.shape[1]



# ========================================
# 20. Numerical preprocessing
# ========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ========================================
# 21. Categorical preprocessing
# ========================================

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ========================================
# 22. Combine preprocessing
# ========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)