import pandas as pd
import matplotlib.pyplot as plt


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