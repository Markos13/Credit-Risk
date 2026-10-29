
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from matplotlib.colors import ListedColormap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


# ========================================
# 1. Project directories
# ========================================

FIGURES_DIR = Path("reports/figures")
MODELS_DIR = Path("models")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# 2. Load dataset
# ========================================

df = pd.read_csv(
    "loan_2014_18.csv",
    low_memory=False
)

print("Original dataset shape:", df.shape)


# ========================================
# 3. Parse loan issuance dates
# ========================================

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
)

df["issue_year"] = df["issue_d"].dt.year

print("\nMissing issuance dates:", df["issue_d"].isna().sum())


# ========================================
# 4. Define target variable
# ========================================

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
}).astype(int)


# ========================================
# 5. Define leakage exclusions
# ========================================

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
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "loan_status",
    "default",
    "id",
    "url",
    "Unnamed: 0",
    "pymnt_plan",
    "funded_amnt_inv"
]


# ========================================
# 6. Additional feature exclusions
# ========================================

additional_exclusions = [
    "funded_amnt",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "initial_list_status",
    "verification_status",
    "verification_status_joint",
    "issue_d",
    "issue_year",
    "emp_title",
    "title",
    "zip_code",
    "policy_code",
    "term",
    "application_type"
]


# ========================================
# 7. Construct model feature dataset
# ========================================

df_model = df_clean.drop(
    columns=leakage_cols + additional_exclusions,
    errors="ignore"
).copy()



# ========================================
# 9. Select eligible loans
# ========================================

eligible_mask = (
    (df_clean["term"].str.strip() == "36 months")
    & (df_clean["issue_year"].between(2014, 2016))
)

df_selected = df_clean.loc[eligible_mask].copy()

X = df_model.loc[df_selected.index].copy()
y = df_selected["default"].copy()


# ========================================
# 10. Chronological train/validation/test split
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

print("\n=== Dataset Splits ===")
print("Training (2014):", X_train.shape)
print("Validation (2015):", X_val.shape)
print("Testing (2016):", X_test.shape)

assert len(X_train) > 0
assert len(X_val) > 0
assert len(X_test) > 0

assert X_train.index.intersection(X_val.index).empty
assert X_train.index.intersection(X_test.index).empty
assert X_val.index.intersection(X_test.index).empty


# ========================================
# 11. Missing-value feature selection
# ========================================

# Determine missingness from 2014 training data only.

missing_pct = X_train.isna().mean() * 100

excluded_features = missing_pct[
    missing_pct > 30
].sort_values(ascending=False)

features_to_keep = missing_pct[
    missing_pct <= 30
].index

X_train = X_train[features_to_keep].copy()
X_val = X_val[features_to_keep].copy()
X_test = X_test[features_to_keep].copy()

print("\nFeatures removed due to >30% missingness:",
      len(excluded_features))


# ========================================
# 12. Feature engineering
# ========================================

for dataset in [X_train, X_val, X_test]:

    # Original credit-line opening date.
    credit_date = pd.to_datetime(
        dataset["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce"
    )

    # Original issuance date, aligned by index.
    issue_date = df_clean.loc[dataset.index, "issue_d"]

    # Credit history duration in months.
    dataset["credit_history_months"] = (
        (issue_date.dt.year - credit_date.dt.year) * 12
        + (issue_date.dt.month - credit_date.dt.month)
    )

    del dataset["earliest_cr_line"]

    # Convert revolving utilization percentages to numeric.
    dataset["revol_util"] = pd.to_numeric(
        dataset["revol_util"]
        .astype("string")
        .str.replace("%", "", regex=False),
        errors="coerce"
    )


# ========================================
# 13. Remove redundant features
# ========================================

redundant_features = [
    "fico_range_high",
    "num_sats",
    "num_rev_tl_bal_gt_0"
]

X_train = X_train.drop(columns=redundant_features)
X_val = X_val.drop(columns=redundant_features)
X_test = X_test.drop(columns=redundant_features)


# ========================================
# 14. Identify numeric/categorical features
# ========================================

numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

assert (
    len(numeric_cols) + len(categorical_cols)
    == X_train.shape[1]
)

assert X_train.columns.equals(X_val.columns)
assert X_train.columns.equals(X_test.columns)

print("\nNumeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))


# ========================================
# 15. Check constant training features
# ========================================

unique_counts = X_train.nunique(dropna=False)

constant_features = unique_counts[
    unique_counts <= 1
]

print("\nConstant training features:",
      len(constant_features))

if len(constant_features) > 0:
    print(constant_features.to_string())


# ========================================
# 16. Preprocessing pipelines
# ========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# ========================================
# 17. Fit preprocessing on training data
# ========================================

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

print("\nProcessed training shape:",
      X_train_processed.shape)

print("Processed validation shape:",
      X_val_processed.shape)


# ========================================
# 18. Train baseline logistic regression
# ========================================

log_reg = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    random_state=42
)

log_reg.fit(X_train_processed, y_train)

y_val_proba = log_reg.predict_proba(
    X_val_processed
)[:, 1]


# ========================================
# 19. Validation probability metrics
# ========================================

print("\n=== Logistic Regression Validation (2015) ===")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_val, y_val_proba):.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision_score(y_val, y_val_proba):.4f}"
)

print(
    f"Brier Score: "
    f"{brier_score_loss(y_val, y_val_proba):.4f}"
)

print(
    f"Log Loss: "
    f"{log_loss(y_val, y_val_proba):.4f}"
)

print(f"Actual default rate: {y_val.mean():.4f}")
print(f"Mean predicted default: {y_val_proba.mean():.4f}")


# ========================================
# 20. Validation classification
# ========================================

threshold = 0.15

y_val_pred = (y_val_proba >= threshold).astype(int)

cm = confusion_matrix(
    y_val,
    y_val_pred,
    labels=[0, 1]
)

print("\n=== Validation Confusion Matrix (15%) ===")
print(cm)

print("\n=== Validation Classification Report ===")

print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=["Fully Paid", "Default"],
        zero_division=0
    )
)


# ========================================
# 21. Save validation confusion matrix
# ========================================

fig, ax = plt.subplots(figsize=(8, 6))

colors = ListedColormap([
    "#B7E4C7",
    "#FFD6A5",
    "#FFADAD",
    "#A0C4FF"
])

color_positions = [
    [0, 1],
    [2, 3]
]

ax.imshow(
    color_positions,
    cmap=colors,
    vmin=0,
    vmax=3
)

labels = [
    ["TN", "FP"],
    ["FN", "TP"]
]

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{labels[i][j]}\n{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="black"
        )

ax.set_xticks([0, 1])
ax.set_xticklabels(["Fully Paid", "Default"])

ax.set_yticks([0, 1])
ax.set_yticklabels(["Fully Paid", "Default"])

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Logistic Regression Confusion Matrix - 2015")

ax.set_xticks([0.5], minor=True)
ax.set_yticks([0.5], minor=True)

ax.grid(
    which="minor",
    color="white",
    linewidth=3
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "confusion_matrix_2015.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ========================================
# 22. Validation threshold comparison
# ========================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

results = []

for current_threshold in thresholds:

    predictions = (
        y_val_proba >= current_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": current_threshold,
        "Precision": precision_score(
            y_val,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_val,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_val,
            predictions,
            zero_division=0
        ),
        "False Positives": fp,
        "False Negatives": fn,
        "Loans Flagged (%)": predictions.mean() * 100
    })

threshold_results = pd.DataFrame(results)

print("\n=== Validation Threshold Comparison ===")
print(
    threshold_results.round(4).to_string(index=False)
)



# ========================================
# 23. Logistic regression coefficients
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

print("\n=== Top 20 Logistic Regression Coefficients ===")

print(
    coefficients.head(20).to_string(index=False)
)


# ========================================
# 24. Combine 2014 and 2015 for final fitting
# ========================================

X_train_final = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_final = pd.concat(
    [y_train, y_val],
    axis=0
)

print("\nFinal training shape:",
      X_train_final.shape)


# ========================================
# 25. Fit final pipeline
# ========================================

final_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])

final_pipeline.fit(
    X_train_final,
    y_train_final
)

print("\nFinal model training completed.")


# ========================================
# 26. Evaluate on 2016 test data
# ========================================

y_test_proba = final_pipeline.predict_proba(
    X_test
)[:, 1]

print("\n=== Final Logistic Regression Test Results (2016) ===")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, y_test_proba):.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision_score(y_test, y_test_proba):.4f}"
)

print(
    f"Brier Score: "
    f"{brier_score_loss(y_test, y_test_proba):.4f}"
)

print(
    f"Log Loss: "
    f"{log_loss(y_test, y_test_proba):.4f}"
)

print(f"\nActual default rate: {y_test.mean():.4f}")
print(f"Mean predicted default: {y_test_proba.mean():.4f}")


# ========================================
# 27. Final test classification at 15%
# ========================================

final_threshold = 0.15

y_test_pred = (
    y_test_proba >= final_threshold
).astype(int)

print("\n=== Test Confusion Matrix (15%) ===")

print(
    confusion_matrix(
        y_test,
        y_test_pred,
        labels=[0, 1]
    )
)

print("\n=== Test Classification Report ===")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Fully Paid", "Default"],
        zero_division=0
    )
)



# ========================================
# 28. Final consistency checks
# ========================================

assert X_train_final.columns.equals(X_test.columns), (
    "Final training and test columns differ."
)

assert len(y_test_proba) == len(y_test)

assert np.isfinite(y_test_proba).all(), (
    "Test probabilities contain non-finite values."
)

assert (
    (y_test_proba >= 0)
    & (y_test_proba <= 1)
).all()

print("\nFinal consistency checks passed.")


# ========================================
# 29. Save final model
# ========================================

joblib.dump(
    {
        "pipeline": final_pipeline,
        "threshold": final_threshold,
        "target": "default",
        "training_years": [2014, 2015],
        "test_year": 2016,
        "feature_columns": X_train_final.columns.tolist()
    },
    MODELS_DIR / "logistic_regression_final.joblib"
)

print("\nFinal model saved successfully.")

print("\nAll analysis steps completed.")
