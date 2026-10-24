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



# 5. Print excluded features
#print("\nFeatures excluded due to >30% missing values:")
#rint(excluded_features.to_string())

#print("\nNumber of features excluded:", len(excluded_features))

# 6. Print dataset shapes
#print("\nDataset shape before filtering:", df_model.shape)
#print("Dataset shape after filtering:", df_filtered.shape)


#print("Features (X):", X.shape)
#print("Target (y):", y.shape)


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

