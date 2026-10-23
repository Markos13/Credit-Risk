import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "loan_2014_18.csv",
    low_memory=False
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

print('hello')