import pandas as pd

df_new = pd.read_csv(
    "loan_2019_20.csv",
    low_memory=False
)

print("Shape:", df_new.shape)

print("\nIssue dates:")
print(df_new["issue_d"].value_counts().head(30))

print("\nLoan statuses:")
print(df_new["loan_status"].value_counts(dropna=False))

print("\nLoan terms:")
print(df_new["term"].value_counts(dropna=False))

print("\nAvailable columns:")
print(df_new.columns.tolist())

# Issue year
df_new["issue_year"] = pd.to_datetime(
    df_new["issue_d"],
    format="%Y-%m-%d",
    errors="coerce"
).dt.year


print("\n=== Loans by Issue Year ===")
print(df_new["issue_year"].value_counts().sort_index())

print("\n=== Loan Status by Issue Year ===")
print(pd.crosstab(
    df_new["issue_year"],
    df_new["loan_status"]
))

print("\n=== 36-Month Loans: Status by Issue Year ===")
df_36 = df_new[
    df_new["term"].str.strip() == "36 months"
]

print(pd.crosstab(
    df_36["issue_year"],
    df_36["loan_status"]
))