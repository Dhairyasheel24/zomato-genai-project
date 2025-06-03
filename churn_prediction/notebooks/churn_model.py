import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# If using the synthetic dataset
df = pd.read_csv('C:\\Users\\HP\\Desktop\\zomato-genai-project\\zomato_churn_synthetic.csv')
print(df.head())

df['membership_status'] = df['membership_status'].fillna('None')


# Check for nulls
print(df.isnull().sum())

# --- 4. Churn Distribution Plot ---
plt.figure()
sns.countplot(x='churned', data=df, palette='Set2')
plt.title("Churn Distribution")
plt.xlabel("Churned (1 = Yes, 0 = No)")
plt.ylabel("Number of Users")
plt.savefig("C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\reports\\churn_distribution.png", bbox_inches='tight')
plt.show()

# --- 5. Feature vs Target Boxplot ---
plt.figure()
sns.boxplot(x='churned', y='last_order_days_ago', data=df, palette='Set3')
plt.title("Last Order Days vs Churn")
plt.xlabel("Churned")
plt.ylabel("Days Since Last Order")
plt.savefig("C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\reports\\last_order_vs_churn.png", bbox_inches='tight')
plt.show()


# --- 6. Encode Categorical Variables for Correlation Matrix ---
df_encoded = pd.get_dummies(df.drop(['user_id'], axis=1), drop_first=True)

# --- 7. Compute Correlation Matrix ---
corr_matrix = df_encoded.corr()

# --- 8. Plot and Save Heatmap ---
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Feature Correlation Heatmap (Including Churn)")
plt.tight_layout()
plt.savefig("C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\reports\\feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# --- 1. Encode features again (if not already done) ---
df_encoded = pd.get_dummies(df.drop(['user_id'], axis=1), drop_first=True)

# --- 2. Define X and y ---
X = df_encoded.drop(['churned'], axis=1)
y = df_encoded['churned']

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Logistic Regression Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 5. Evaluate Model ---
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# --- 6. Identify Top-5 Likely Churners ---
y_probs = model.predict_proba(X_test)[:, 1]

X_test_copy = X_test.copy()
X_test_copy['prob_churn'] = y_probs

# Recover user_id from original df
X_test_copy['user_id'] = df.loc[X_test.index, 'user_id']

# Top 5 users most likely to churn
# Visual bar chart of top 5 churners
plt.figure(figsize=(8, 5))
top_churners = X_test_copy.sort_values(by='prob_churn', ascending=False).head(5)
sns.barplot(x='prob_churn', y='user_id', data=top_churners, palette='Reds_r')
plt.xlabel("Churn Probability")
plt.ylabel("User ID")
plt.title("Top 5 Users Most Likely to Churn")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\reports\\top5_churn_risk_users.png", dpi=300)
plt.show()
