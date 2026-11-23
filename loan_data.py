
"""Deterministic sample bank-loan dataset used when no trainer dataset is supplied."""
import numpy as np
import pandas as pd

def make_loan_data(n_samples=1200, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 71, n_samples)
    income = np.clip(rng.normal(65000, 22000, n_samples), 18000, 180000)
    employment_years = np.clip(rng.normal(8, 6, n_samples), 0, 40)
    credit_score = np.clip(rng.normal(680, 65, n_samples), 420, 850)
    loan_amount = np.clip(rng.normal(18000, 9000, n_samples), 1000, 60000)
    debt_to_income = np.clip(rng.beta(2.2, 5, n_samples), 0.01, 0.95)
    employment_type = rng.choice(["Salaried", "Self-employed", "Contract"], n_samples, p=[.62,.25,.13])
    education = rng.choice(["High School", "Graduate", "Postgraduate"], n_samples, p=[.30,.48,.22])
    purpose = rng.choice(["Home", "Car", "Education", "Personal", "Business"], n_samples)
    grade = rng.choice(["A", "B", "C", "D", "E"], n_samples, p=[.18,.30,.28,.17,.07])
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n_samples), unit="D")
    grade_risk = pd.Series(grade).map({"A":-1.0,"B":-.4,"C":0.2,"D":.8,"E":1.3}).to_numpy()
    emp_risk = pd.Series(employment_type).map({"Salaried":-.25,"Self-employed":.25,"Contract":.55}).to_numpy()
    logit = (-2.0 + 0.000025*(65000-income) + 0.010*(680-credit_score)
             + 2.4*debt_to_income + 0.000035*(loan_amount-18000)
             - 0.025*employment_years + grade_risk*.55 + emp_risk*.35)
    probability = 1/(1+np.exp(-logit))
    default = rng.binomial(1, probability)
    # Add realistic missing values to raw fields
    df = pd.DataFrame({
        "age": age, "income": income.round(2), "employment_years": employment_years.round(1),
        "credit_score": credit_score.round(0), "loan_amount": loan_amount.round(2),
        "debt_to_income": debt_to_income.round(3), "employment_type": employment_type,
        "education": education, "purpose": purpose, "grade": grade,
        "application_date": dates.astype(str), "default": default
    })
    for col, frac in [("income",.04),("credit_score",.03),("employment_type",.03),("education",.02)]:
        idx = rng.choice(n_samples, int(n_samples*frac), replace=False)
        df.loc[idx, col] = np.nan
    return df
