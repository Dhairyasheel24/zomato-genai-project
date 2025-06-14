import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="Zomato GenAI Churn Dashboard", layout="wide")

st.title("🍽️ Zomato GenAI Churn Dashboard")
st.markdown("---")

# Sidebar Branding
st.sidebar.image("churn_prediction/assets/zomato_logo.png", width=150)
st.sidebar.title("Zomato GenAI Churn Lab")

# --- Load fixed Zomato dataset ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

data_path = "churn_prediction/data/zomato_churn_synthetic.csv"
df = load_csv(data_path)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

if 'churned' not in df.columns:
    st.error("Dataset must contain 'churned' column.")
else:
    # --- Preprocessing ---
    df['membership_status'] = df['membership_status'].fillna('None')
    df_encoded = pd.get_dummies(df.drop(['user_id'], axis=1), drop_first=True)
    X = df_encoded.drop('churned', axis=1)
    y = df_encoded['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Options ---
    model_option = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"])

    def train_model(X_train, y_train, model_name):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        return model

    # --- Train Model ---
    model = train_model(X_train, y_train, model_option)
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    st.markdown("---")
    st.subheader("📈 Evaluation Metrics")
    st.code(classification_report(y_test, y_pred))
    st.text("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, probs)))

    # --- Top 5 churners ---
    top5_df = X_test.copy()
    top5_df['user_id'] = df.loc[X_test.index, 'user_id']
    top5_df['prob_churn'] = probs
    top5_sorted = top5_df.sort_values(by='prob_churn', ascending=False).head(5)

    st.markdown("---")
    st.subheader("🔍 Top 5 High-Risk Users")
    st.dataframe(top5_sorted[['user_id', 'prob_churn']], use_container_width=True)

    fig = px.bar(top5_sorted, x='prob_churn', y='user_id', orientation='h',
                 title='Top 5 Users Most Likely to Churn', color='prob_churn')
    st.plotly_chart(fig, use_container_width=True)

    # --- Save top5 for GPT use ---
    top5_sorted[['user_id', 'prob_churn']].to_csv("churn_prediction/data/top5_churn_risk_users.csv", index=False)

    # --- GPT Offer Section ---
    st.markdown("---")
    st.header("🤖 GPT-Generated Retention Messages")
    try:
        offer_df = load_csv("churn_prediction/data/top5_with_gpt_offers.csv")
        st.dataframe(offer_df[['user_id', 'prob_churn', 'gpt_offer']], use_container_width=True)

        st.subheader("📩 Personalized Messages")
        for _, row in offer_df.iterrows():
            st.markdown(f"""
            **👤 {row['user_id']} ({row['prob_churn']:.2%})**  
            💬 {row['gpt_offer']}
            """)
    except:
        st.info("⚠️ GPT offer file not found. Run the offer generator script to generate personalized messages.")
