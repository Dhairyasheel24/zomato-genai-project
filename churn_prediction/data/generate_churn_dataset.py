import pandas as pd
import numpy as np
import random

np.random.seed(42)

n_users = 1000
data = {
    "user_id": [f"U{1000+i}" for i in range(n_users)],
    "last_order_days_ago": np.random.randint(1, 90, n_users),
    "orders_last_30d": np.random.poisson(lam=3, size=n_users),
    "avg_order_value": np.round(np.random.normal(loc=350, scale=50, size=n_users), 2),
    "support_tickets": np.random.poisson(lam=0.5, size=n_users),
    "app_opens_last_30d": np.random.randint(0, 100, n_users),
    "payment_method": np.random.choice(["UPI", "Card", "Cash"], size=n_users, p=[0.5, 0.3, 0.2]),
    "membership_status": np.random.choice(["Pro", "None"], size=n_users, p=[0.3, 0.7]),
    "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=n_users, p=[0.4, 0.4, 0.2])
}

# Generate churn label (more likely to churn if app usage is low or no Pro)
def churn_logic(row):
    score = 0
    if row["last_order_days_ago"] > 60: score += 2
    if row["orders_last_30d"] == 0: score += 2
    if row["membership_status"] == "None": score += 1
    if row["app_opens_last_30d"] < 10: score += 2
    if row["support_tickets"] > 1: score += 1
    return 1 if score >= 4 else 0

df = pd.DataFrame(data)
df["churned"] = df.apply(churn_logic, axis=1)

df.to_csv("zomato_churn_synthetic.csv", index=False)
print("✅ Dataset generated: zomato_churn_synthetic.csv")
