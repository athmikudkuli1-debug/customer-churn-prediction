"""
STEP 1: Generate a realistic customer dataset

In a real project you'd load this from a CSV a company gives you
(e.g. the popular "Telco Customer Churn" dataset on Kaggle).
Here we SIMULATE one so you have real data to practice on, with
built-in realistic patterns (e.g. month-to-month contracts churn
more, long-tenure customers churn less) so the models we train
later actually find meaningful signal.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # makes results reproducible every time we run this

N = 2000  # number of customers

# --- Basic demographic / account features ---
gender = np.random.choice(["Male", "Female"], size=N)
senior_citizen = np.random.choice([0, 1], size=N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=N, p=[0.30, 0.70])

# tenure = how many months they've been a customer
tenure = np.random.exponential(scale=24, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.24, 0.21]
)
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], size=N, p=[0.34, 0.44, 0.22]
)
online_security = np.random.choice(["Yes", "No"], size=N, p=[0.29, 0.71])
tech_support = np.random.choice(["Yes", "No"], size=N, p=[0.29, 0.71])
paperless_billing = np.random.choice(["Yes", "No"], size=N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=N, p=[0.34, 0.23, 0.22, 0.21],
)

monthly_charges = np.round(np.random.normal(65, 30, size=N).clip(18, 120), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, size=N), 2)
total_charges = total_charges.clip(0, None)

# --- Build churn probability from realistic risk factors ---
# This is the "ground truth" logic — in real life you don't know this,
# the model has to discover it from the data. We use it here just to
# make the simulated data behave realistically.
churn_prob = np.full(N, 0.03)
churn_prob += (contract == "Month-to-month") * 0.18
churn_prob += (contract == "One year") * 0.03
churn_prob += (internet_service == "Fiber optic") * 0.06
churn_prob += (tech_support == "No") * 0.07
churn_prob += (online_security == "No") * 0.05
churn_prob += (payment_method == "Electronic check") * 0.06
churn_prob += (tenure < 6) * 0.12
churn_prob -= (tenure > 48) * 0.08
churn_prob += (monthly_charges > 90) * 0.05
churn_prob += (senior_citizen == 1) * 0.03
churn_prob -= (partner == "Yes") * 0.03
churn_prob = churn_prob.clip(0.02, 0.85)

churn = np.random.binomial(1, churn_prob)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(N)],
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "Contract": contract,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

out_path = "customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Churn"].value_counts(normalize=True))
