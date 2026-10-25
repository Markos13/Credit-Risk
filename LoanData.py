import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


df = pd.read_csv(
    "loan_2014_18.csv"
)


df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
)


df["issue_year"] = df["issue_d"].dt.year

print("Missing issuance dates:", df["issue_d"].isna().sum())

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


# Remove leakage features and identifiers
df_model = df_clean.drop(
    columns=leakage_cols,
    errors="ignore"
).copy()




# Calculate yearly loan statistics
yearly_summary = df_clean.groupby("issue_year")["default"].agg(
    total_loans="count",
    total_defaults="sum",
    default_rate="mean"
)

yearly_summary["default_rate"] *= 100


print("\nYearly default statistics:")
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


print("\nLoan term distribution:")
print(term_distribution.round(2))


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


# Select 36-month loans issued between 2014 and 2016
eligible_mask = (
    (df_clean["term"].str.strip() == "36 months") &
    (df_clean["issue_year"].between(2014, 2016))
)

# Keep the selected loans
df_selected = df_clean.loc[eligible_mask].copy()

# Select corresponding model features
X = df_model.loc[df_selected.index].copy()

# Select corresponding target
y = df_selected["default"].copy()

'''print("Selected loans:", len(df_selected))
print("Features:", X.shape)
print("Target:", y.shape)'''


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

print("\nDataset shapes:")

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
print("Validation / Test:", len(X_val.index.intersection(X_test.index)))