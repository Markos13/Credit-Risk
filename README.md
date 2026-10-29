# Lending Club Loan Default Prediction

### Credit Risk Analysis Using Python and Logistic Regression

## Project Overview

This project develops a machine learning model to estimate the probability of loan default using historical Lending Club loan data.

The objective is to examine how borrower and loan characteristics relate to credit risk and build an interpretable model that distinguishes between loans that are fully repaid and loans that default.

The project covers an end-to-end machine learning workflow, including data cleaning, feature engineering, leakage-aware feature selection, chronological data splitting, preprocessing, model training, evaluation, and model serialization.

**Final model:** Logistic Regression
**Programming language:** Python
**Data period:** 2014–2016
**Prediction target:** Eventual observed loan default

## Business Problem

Credit risk assessment is an important part of consumer lending. Lenders need to estimate the likelihood of borrowers failing to repay their loans to support underwriting, pricing, and portfolio risk management.

This project addresses the following question:

**Can borrower and loan characteristics be used to estimate the probability of eventual loan default?**

The analysis focuses on both risk ranking and predicted default probabilities rather than classification accuracy alone.

## Dataset

The project uses historical Lending Club loan records from `loan_2014_18.csv`.

The modeling dataset includes:

- Loans issued between 2014 and 2016
- Loans with a 36-month term
- Loans with a resolved status of `Fully Paid`, `Charged Off`, or `Default`

The binary target is defined as follows:

| Loan Status | Target |
| ----------- | ------ |
| Fully Paid  | 0      |
| Charged Off | 1      |
| Default     | 1      |

Unresolved loans, including those with a `Current` status, are excluded.

### Chronological Data Split

Rather than randomly splitting the observations, the project separates loans according to their issuance year.

| Dataset    | Issuance Year |       Loans |
| ---------- | ------------- | ----------: |
| Training   | 2014          |     162,570 |
| Validation | 2015          |     283,173 |
| Testing    | 2016          |     323,436 |
| **Total**  |               | **769,179** |

This approach evaluates the model on a later issuance cohort and avoids randomly mixing loan vintages.

The evaluation remains retrospective because final observed repayment outcomes are used to construct the modeling dataset.

## Methodology

### 1. Data Cleaning and Target Construction

Loan issuance dates are parsed, eligible 36-month loans are selected, and resolved loan statuses are converted into a binary default target.

### 2. Data Leakage Prevention

Features containing information generated after loan issuance are excluded.

Examples include:

- Total payments received
- Principal recovered
- Recovery amounts
- Outstanding principal
- Last payment information
- Loan status
- Hardship and settlement information

Additional variables, including interest rate, installment, grade, and subgrade, are excluded from the chosen model specification.

These exclusions aim to reduce target leakage and focus the model on borrower and loan characteristics.

### 3. Missing-Value Handling

Features with more than 30% missing values in the 2014 training cohort are removed.

**33 features are excluded** using this rule.

Remaining missing values are handled through preprocessing:

- Numerical features: median imputation
- Categorical features: most-frequent-value imputation

Missingness-based feature selection is performed using the training cohort only.

### 4. Feature Engineering

The project constructs `credit_history_months` from the difference between loan issuance date and the borrower's earliest recorded credit line.

The `revol_util` variable is converted from percentage-formatted text into a numerical feature.

Redundant features are also removed.

The final feature set contains **50 numerical and 4 categorical features**, before one-hot encoding.

### 5. Preprocessing

Preprocessing is implemented with scikit-learn pipelines.

Numerical features undergo median imputation and standard scaling.

Categorical features undergo most-frequent imputation and one-hot encoding.

Preprocessing is initially fitted using 2014 training data and applied to the 2015 validation cohort.

For the final model, preprocessing is refitted on the combined 2014–2015 dataset.

### 6. Model Training

The project uses Logistic Regression with the following configuration:

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
```

Logistic Regression is selected for its simplicity, interpretability, and suitability as a baseline for binary credit-risk modeling.

### 7. Validation and Classification Threshold

The initial model is trained on 2014 loans and evaluated on 2015 loans.

Model performance is assessed using ROC-AUC, Average Precision, Brier Score, Log Loss, and classification metrics.

A **15% classification threshold** is used to classify loans:

- Probability below 15%: predicted non-default
- Probability of at least 15%: predicted default

The validation analysis also compares alternative thresholds to illustrate the precision-recall trade-off.

### 8. Final Model Training

The final Logistic Regression pipeline is trained using the combined 2014 and 2015 cohorts.

The 2016 cohort is used for the final held-out evaluation.

## Model Results

### Final Test Performance — 2016

| Metric                            |     Result |
| --------------------------------- | ---------: |
| ROC-AUC                           | **0.6776** |
| Average Precision                 | **0.2728** |
| Brier Score                       | **0.1268** |
| Log Loss                          | **0.4128** |
| Default Precision (15% threshold) |     \~0.24 |
| Default Recall (15% threshold)    |     \~0.62 |
| Default F1-score (15% threshold)  |     \~0.35 |

### Results Interpretation

**ROC-AUC: 0.6776**

The model demonstrates moderate ability to distinguish between loans that eventually default and loans that are fully repaid.

**Average Precision: 0.2728**

Average Precision summarizes the precision-recall relationship and is particularly useful when defaults are less common than non-defaults.

**Brier Score: 0.1268**

The Brier Score measures the squared error of predicted default probabilities. Lower scores indicate better probability predictions.

**Default Recall: approximately 62%**

At the selected threshold, the model identifies approximately 62% of actual defaults.

**Default Precision: approximately 24%**

Approximately 24% of loans flagged as defaults actually default.

These results highlight the trade-off between detecting more defaults and incorrectly flagging loans that are ultimately repaid.

The model provides moderate predictive value but is not intended for automated lending decisions.

## Visualizations

The analysis produces a validation confusion matrix at the 15% threshold.

The figure is saved as:

`reports/figures/confusion_matrix_2015.png`

The script also reports a validation threshold comparison and the largest Logistic Regression coefficients.

## Technologies Used

- **Python** — programming language
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **scikit-learn** — preprocessing, modeling, and evaluation
- **Matplotlib** — visualization
- **joblib** — model serialization

Project dependencies are specified in `requirements.txt`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Markos13/Credit-Risk.git
cd Credit-Risk
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required Python packages using the repository's `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the Lending Club dataset in the repository's root directory with the filename:

```text
loan_2014_18.csv
```

The dataset may need to be obtained separately if it is not included in the repository.

### 5. Run the analysis

Run the project's main Python analysis script from the repository root.

The script performs data preparation, model training, validation, final evaluation, and model saving.

## Saved Model

The final trained Logistic Regression model is saved as:

`models/logistic_regression_final.joblib`

The saved artifact contains:

- The fitted preprocessing and classification pipeline
- The selected classification threshold
- Target variable name
- Training and test years
- Expected feature column names

**Note:** The saved pipeline expects pre-engineered input features matching the training schema. It does not directly accept unprocessed loan records.

## Limitations

**Resolved-loan selection bias:** The dataset excludes loans with unresolved outcomes, potentially reducing its representativeness of all loans at origination.

**Retrospective evaluation:** The chronological split uses issuance years, but final loan outcomes are observed retrospectively. This is not a real-time historical backtest with a fixed outcome horizon.

**Moderate predictive performance:** The model provides useful risk differentiation but cannot reliably identify every default.

**False positives:** The 15% classification threshold captures more defaults at the cost of flagging many loans that are ultimately repaid.

**Feature availability:** Retained variables should be verified as available at origination before any real-world application.

**Generalizability:** Results may not generalize to other lenders, loan vintages, or economic conditions.

**No financial optimization:** The classification threshold is evaluated using statistical metrics rather than expected financial losses or lending profitability.

## Potential Future Improvements

- More detailed probability calibration analysis
- Stability testing across issuance cohorts
- Evaluation using a fixed default observation horizon
- Expected credit loss modeling
- Additional model interpretability analysis
- A reusable feature-engineering pipeline for raw loan records

## Conclusion

This project demonstrates an end-to-end credit-risk modeling workflow using historical Lending Club data and Logistic Regression.

The final model achieves a **ROC-AUC of 0.6776** and **Average Precision of 0.2728** on the 2016 test cohort, demonstrating moderate ability to distinguish loans that default from loans that are fully repaid.

Beyond predictive performance, the project emphasizes important machine learning practices, including chronological data splitting, leakage-aware feature selection, training-only preprocessing, validation-based threshold assessment, and evaluation on a held-out cohort.
