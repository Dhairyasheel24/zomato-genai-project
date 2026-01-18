import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

def generate_data(n_users=2000):
    # --- 1. Generate Feature Data ---
    data = {
        "user_id": [f"U{1000+i}" for i in range(n_users)],
        "last_order_days_ago": np.random.randint(1, 90, n_users),
        "orders_last_30d": np.random.poisson(lam=3, size=n_users),
        "avg_order_value": np.round(np.random.normal(loc=350, scale=100, size=n_users), 2),
        "support_tickets": np.random.choice([0, 1, 2, 3], size=n_users, p=[0.7, 0.2, 0.08, 0.02]),
        "app_opens_last_30d": np.random.randint(0, 60, n_users),
        "payment_method": np.random.choice(["UPI", "Card", "Cash"], size=n_users, p=[0.6, 0.3, 0.1]),
        "membership_status": np.random.choice(["Pro", "None"], size=n_users, p=[0.2, 0.8]),
        "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=n_users, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # --- 2. Probabilistic Churn Generation ---
    # We want a base churn rate of approx 20-25% (Imbalanced Class)
    
    # Intercept: Lower value = Lower base probability
    logit = -2.5 
    
    # Feature weights (adjusted for realism)
    logit += 0.06 * df["last_order_days_ago"]          # Strongest driver
    logit -= 0.25 * df["orders_last_30d"]              # Active users stay
    logit += 0.90 * df["support_tickets"]              # Complaints = risk
    logit -= 0.001 * df["avg_order_value"]             # High spender = slightly loyal
    logit -= 0.60 * (df["membership_status"] == "Pro") # Pro members sticky
    logit += 0.30 * (df["payment_method"] == "Cash")   # Cash users flake
    
    # Add Gaussian Noise (Simulates real-world unpredictability)
    noise = np.random.normal(0, 1.0, n_users)
    logit += noise
    
    # Sigmoid function to get probability
    df["churn_probability"] = 1 / (1 + np.exp(-logit))
    
    # Bernoulli Trial
    df["churned"] = np.random.binomial(n=1, p=df["churn_probability"])
    
    # Clean up
    df = df.drop(columns=["churn_probability"])
    
    # Stats
    churn_rate = df["churned"].mean()
    print(f"Dataset Generated with Churn Rate: {churn_rate:.2%}")
    print("(Targeting 20-25% for realistic imbalance)")
    
    return df

if __name__ == "__main__":
    df = generate_data()
    # Ensure this path matches your folder structure
    df.to_csv("churn_prediction/data/zomato_churn_synthetic.csv", index=False)
    print("✅ Data saved to churn_prediction/data/zomato_churn_synthetic.csv")