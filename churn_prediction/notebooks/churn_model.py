import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# --- 1. Load Data ---
# Note: Using 'churn_prediction/data/' path based on your logs
df = pd.read_csv('churn_prediction/data/zomato_churn_synthetic.csv')

X = df.drop(['user_id', 'churned'], axis=1)
y = df['churned']

# --- 2. Preprocessing ---
cat_cols = ['payment_method', 'membership_status', 'city_tier']
num_cols = ['last_order_days_ago', 'orders_last_30d', 'avg_order_value', 'support_tickets', 'app_opens_last_30d']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

# Stratified Split (Crucial for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Explainability (Logistic Regression) ---
print("\n--- 🔍 Logistic Regression (Key Drivers) ---")
log_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))])
log_model.fit(X_train, y_train)

# Show coefficients
feature_names = (num_cols + 
                 list(log_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)))
coeffs = log_model.named_steps['classifier'].coef_[0]
coeff_df = pd.DataFrame({'Feature': feature_names, 'Weight': coeffs})
print(coeff_df.sort_values(by='Weight', ascending=False).head(5))

# --- 4. Performance (XGBoost) ---
print("\n--- 🚀 XGBoost (Prediction) ---")
# Calculate scale_pos_weight dynamically
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', XGBClassifier(
                                n_estimators=100,
                                max_depth=3,            # Prevent overfitting
                                learning_rate=0.1,
                                scale_pos_weight=scale_pos_weight, # Handle Imbalance
                                eval_metric='logloss',
                                random_state=42
                            ))])

xgb_model.fit(X_train, y_train)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.3f}")

# --- 5. Business Threshold Tuning ---
def calculate_profit(threshold):
    # Assumptions: LTV=500, Coupon Cost=50
    preds = (y_probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    return (tp * 450) - (fp * 50) # 450 = 500 - 50

thresholds = np.arange(0.1, 0.9, 0.05)
profits = [calculate_profit(t) for t in thresholds]
best_threshold = thresholds[np.argmax(profits)]

print(f"💰 Optimal Business Threshold: {best_threshold:.2f}")

# --- 6. Export Top Risk Users ---
full_probs = xgb_model.predict_proba(X)[:, 1]
df['prob_churn'] = full_probs
top_risk = df.sort_values(by='prob_churn', ascending=False).head(5)
top_risk[['user_id', 'prob_churn']].to_csv('churn_prediction/data/top5_churn_risk_users.csv', index=False)
print("✅ Top 5 Risk Users exported.")