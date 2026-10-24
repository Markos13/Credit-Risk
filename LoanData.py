import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


df = pd.read_csv(
    "loan_2014_18.csv"
)

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
    "Unnamed: 0"
]


# Remove leakage features and identifiers
df_model = df_clean.drop(
    columns=leakage_cols,
    errors="ignore"
).copy()


# 2. Calculate missing percentage for each feature
missing_pct = df_model.isnull().mean() * 100


# 3. Identify features with more than 30% missing values
excluded_features = missing_pct[missing_pct > 30].sort_values(
    ascending=False
)


# 4. Keep only features with <=30% missing values
features_to_keep = missing_pct[missing_pct <= 30].index

df_filtered = df_model[features_to_keep].copy()


# 5. Print excluded features
#print("\nFeatures excluded due to >30% missing values:")
#rint(excluded_features.to_string())

#print("\nNumber of features excluded:", len(excluded_features))

# 6. Print dataset shapes
#print("\nDataset shape before filtering:", df_model.shape)
#print("Dataset shape after filtering:", df_filtered.shape)


X = df_filtered.copy()
y = df_clean["default"].copy()

#print("Features (X):", X.shape)
#print("Target (y):", y.shape)


df_dates = pd.read_csv(
    "loan_2014_18.csv",
    usecols=["issue_d"]
)

'''print(df_dates["issue_d"].head(20))
print(df_dates["issue_d"].dtype)
print(df_dates["issue_d"].dropna().unique()[:20])'''


df_clean["issue_d"] = pd.to_datetime(
    df_dates.loc[df_clean.index, "issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
)

#print("Earliest loan:", df_clean["issue_d"].min())
#print("Latest loan:", df_clean["issue_d"].max())


# Extract the year of loan issuance
df_clean["issue_year"] = df_clean["issue_d"].dt.year

# Calculate yearly loan statistics
yearly_summary = df_clean.groupby("issue_year")["default"].agg(
    total_loans="count",
    total_defaults="sum",
    default_rate="mean"
)

yearly_summary["default_rate"] *= 100

print(yearly_summary)



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

#print(term_distribution.round(2))



# Use the original dataset, before filtering statuses
df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
)

df["issue_year"] = df["issue_d"].dt.year

status_by_year = pd.crosstab(
    df["issue_year"],
    df["loan_status"],
    normalize="index"
) * 100

print(status_by_year.round(2).to_string())

'''

# Make sure X and y correspond to the same loans
assert X.index.equals(y.index)

# Get loan issuance dates
issue_dates = df_clean.loc[X.index, "issue_d"]

# Verify dates
#print("Missing issue dates:", issue_dates.isna().sum())

assert issue_dates.notna().all()

# Create chronological masks
train_mask = issue_dates < "2017-01-01"

val_mask = (
    (issue_dates >= "2017-01-01") &
    (issue_dates < "2018-01-01")
)

test_mask = (
    (issue_dates >= "2018-01-01") &
    (issue_dates < "2019-01-01")
)

# Split features
X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()

# Split target
y_train = y.loc[train_mask].copy()
y_val = y.loc[val_mask].copy()
y_test = y.loc[test_mask].copy()

for dataset in [X_train, X_val, X_test]:
    dataset.drop(
        columns=["issue_d", "issue_year"],
        errors="ignore",
        inplace=True
    )


print("TRAINING")
print("Shape:", X_train.shape)
print("Default rate:", round(y_train.mean() * 100, 2), "%")

print("\nVALIDATION")
print("Shape:", X_val.shape)
print("Default rate:", round(y_val.mean() * 100, 2), "%")

print("\nTESTING")
print("Shape:", X_test.shape)
print("Default rate:", round(y_test.mean() * 100, 2), "%")


total_split = len(X_train) + len(X_val) + len(X_test)

print("\nTotal original observations:", len(X))
print("Total observations after splitting:", total_split)
print("Unassigned observations:", len(X) - total_split)


# Search for potentially useful date and outcome columns
date_status_cols = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in [
        "date", "_d", "status", "dpd", "payment"
    ])
]

print(date_status_cols)

print(df["loan_status"].value_counts())
print(df["term"].value_counts())


df["last_pymnt_d"] = pd.to_datetime(
    df["last_pymnt_d"],
    format="%b-%Y",
    errors="coerce"
)

print("Latest payment:", df["last_pymnt_d"].max())

print(
    df.groupby("issue_year")["last_pymnt_d"]
    .max()
)

'''

# Keep only 36-month loans
df_36 = df[
    df["term"].str.strip() == "36 months"
].copy()

# Identify resolved loans
df_36["resolved"] = df_36["loan_status"].isin([
    "Fully Paid",
    "Charged Off",
    "Default"
])

# Calculate resolution rates by issuance year
resolution_summary = df_36.groupby("issue_year")["resolved"].agg(
    total_loans="count",
    resolved_loans="sum",
    resolved_rate="mean"
)

resolution_summary["unresolved_pct"] = (
    100 * (1 - resolution_summary["resolved_rate"])
)

print(resolution_summary.round(2))
