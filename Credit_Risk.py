import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


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
    "term"
    
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



# ========================================
# 23. Fit and transform
# ========================================

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

'''print("\nProcessed dataset shapes:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)

print("\nMissing values after preprocessing:")


print(
    "Training:",
    np.isnan(X_train_processed).sum()
)

print(
    "Validation:",
    np.isnan(X_val_processed).sum()
)'''



# ========================================
# 24. Train logistic regression baseline
# ========================================

log_reg = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    random_state=42
)

log_reg.fit(X_train_processed, y_train)

#print("\nLogistic regression training completed.")
#print("Iterations used:", log_reg.n_iter_[0])



# ========================================
# 25. Generate validation predictions
# ========================================

y_val_proba = log_reg.predict_proba(
    X_val_processed
)[:, 1]

#print("\nFirst 10 predicted default probabilities:")
#print(y_val_proba[:10])


# ========================================
# 26. Evaluate validation performance
# ========================================

roc_auc = roc_auc_score(
    y_val,
    y_val_proba
)

pr_auc = average_precision_score(
    y_val,
    y_val_proba
)

brier = brier_score_loss(
    y_val,
    y_val_proba
)

logloss = log_loss(
    y_val,
    y_val_proba
)

print("\n=== Logistic Regression Validation Results (2015) ===")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Average Precision: {pr_auc:.4f}")
print(f"Brier Score:       {brier:.4f}")
print(f"Log Loss:          {logloss:.4f}")

print(f"\nActual default rate:    {y_val.mean():.4f}")
print(f"Mean predicted default: {y_val_proba.mean():.4f}")



# ========================================
# 28. Confusion matrix
# ========================================

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Convert probabilities into binary predictions
threshold = 0.5

y_val_pred = (y_val_proba >= threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])

# Keep terminal output unchanged
print("\n=== Confusion Matrix (Threshold = 0.50) ===")
print(cm)

# Classification report
print("\n=== Classification Report ===")
print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=["Fully Paid", "Default"],
        zero_division=0
    )
)

# ========================================
# Visualize confusion matrix
# ========================================

fig, ax = plt.subplots(figsize=(8, 6))

# Different color for each cell
colors = ListedColormap([
    "#B7E4C7",  # TN - Green
    "#FFD6A5",  # FP - Orange
    "#FFADAD",  # FN - Red
    "#A0C4FF"   # TP - Blue
])

# Assign colors independently of confusion matrix values
color_positions = [
    [0, 1],
    [2, 3]
]

ax.imshow(color_positions, cmap=colors, vmin=0, vmax=3)

# Cell labels
labels = [
    ["TN", "FP"],
    ["FN", "TP"]
]

for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            f"{labels[i][j]}\n{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="black"
        )

# Axis labels
ax.set_xticks([0, 1])
ax.set_xticklabels(["Fully Paid", "Default"])

ax.set_yticks([0, 1])
ax.set_yticklabels(["Fully Paid", "Default"])

ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)

ax.set_title(
    "Logistic Regression Confusion Matrix - 2015",
    fontsize=15,
    pad=15
)


# Draw borders between cells
ax.set_xticks([0.5], minor=True)
ax.set_yticks([0.5], minor=True)
ax.grid(which="minor", color="white", linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()

# Save image without color bar
plt.savefig(
    "reports/figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)



# ========================================
# 29. Compare classification thresholds
# ========================================

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

results = []

for threshold in thresholds:

    predictions = (y_val_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "False Positives": fp,
        "False Negatives": fn,
        "Loans Flagged (%)": predictions.mean() * 100
    })

threshold_results = pd.DataFrame(results)

print("\n=== Threshold Comparison ===")
print(threshold_results.round(4).to_string(index=False))



# ========================================
# 30. Calibration analysis
# ========================================


fraction_of_positives, mean_predicted_value = calibration_curve(
    y_val,
    y_val_proba,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Default Probability")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Curve - 2015 Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/figures/calibration_curve_2015.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ========================================
# 31. Logistic regression coefficients
# ========================================

feature_names = preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": log_reg.coef_[0]
})

coefficients["abs_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)

'''print("\n=== Top 20 Logistic Regression Coefficients ===")
print(
    coefficients.head(20).to_string(index=False)
)

print("\n=== Top 10 Features Increasing Default Risk ===")
print(
    coefficients.sort_values(
        "coefficient",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\n=== Top 10 Features Decreasing Default Risk ===")
print(
    coefficients.sort_values(
        "coefficient",
        ascending=True
    ).head(10).to_string(index=False)
)'''


print("\nApplication type distribution in training:")
print(X_train["application_type"].value_counts(dropna=False))

print("\nApplication type distribution in validation:")
print(X_val["application_type"].value_counts(dropna=False))

print("\nApplication type categories learned:")
print(
    preprocessor.named_transformers_["cat"]
    .named_steps["encoder"]
    .categories_[categorical_cols.index("application_type")]
)

