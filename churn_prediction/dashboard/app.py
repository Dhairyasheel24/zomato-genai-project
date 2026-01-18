import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score

# --- Page Config ---
st.set_page_config(page_title="Zomato Retention AI", layout="wide", page_icon="🍽️")

# --- Helper Functions ---
@st.cache_data
def load_data():
    # Try multiple paths to find the data
    paths = [
        "churn_prediction/data/zomato_churn_synthetic.csv",
        "../data/zomato_churn_synthetic.csv",
        "C:/Users/HP/Desktop/zomato-genai-project/churn_prediction/data/zomato_churn_synthetic.csv"
    ]
    for p in paths:
        try:
            return pd.read_csv(p)
        except:
            continue
    return None

def train_models(df):
    """
    Trains two models on the fly:
    1. Logistic Regression (for Explainability/Coefficients)
    2. XGBoost (for High Performance Prediction)
    """
    X = df.drop(['user_id', 'churned'], axis=1)
    y = df['churned']
    
    # Identify columns
    cat_cols = ['payment_method', 'membership_status', 'city_tier']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model 1: Logistic Regression (Explainability) ---
    log_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))])
    log_model.fit(X_train, y_train)

    # --- Model 2: XGBoost (Performance) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', XGBClassifier(
                                    n_estimators=100, max_depth=3, learning_rate=0.1, 
                                    scale_pos_weight=scale_pos_weight, random_state=42,
                                    eval_metric='logloss'
                                ))])
    xgb_model.fit(X_train, y_train)
    
    return log_model, xgb_model, X_test, y_test, num_cols, cat_cols

# --- MAIN APP LAYOUT ---
st.title("🍽️ Zomato Customer Retention Engine")
st.markdown("### Interview-Grade Churn Prediction & Intervention Pipeline")

# 1. Load Data
df = load_data()
if df is None:
    st.error("❌ Data not found. Please run 'generate_churn_dataset.py' first.")
    st.stop()

# 2. Train Models (Cached)
if 'models_trained' not in st.session_state:
    with st.spinner('Training Models (Logistic Regression & XGBoost)...'):
        log_model, xgb_model, X_test, y_test, num_cols, cat_cols = train_models(df)
        st.session_state['log_model'] = log_model
        st.session_state['xgb_model'] = xgb_model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['num_cols'] = num_cols
        st.session_state['cat_cols'] = cat_cols
        st.session_state['models_trained'] = True

# Retrieve from session
log_model = st.session_state['log_model']
xgb_model = st.session_state['xgb_model']
X_test = st.session_state['X_test']
y_test = st.session_state['y_test']

# Get Probabilities (XGBoost is the Champion Model)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# --- SIDEBAR: Business Levers ---
st.sidebar.header("⚙️ Business Strategy")
st.sidebar.markdown("Adjust these to see Profit Impact")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.4, 0.05, 
                              help="Probability above which we classify as 'Churn'. Lower = More Aggressive.")
customer_value = st.sidebar.number_input("LTV of Saved Customer (₹)", value=500, step=50)
coupon_cost = st.sidebar.number_input("Cost of Retention Offer (₹)", value=50, step=10)

# --- KEY METRICS ROW ---
preds = (y_probs >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

saved_revenue = tp * (customer_value - coupon_cost)
wasted_spend = fp * coupon_cost
# net_profit = Revenue Saved - Cost of Wasted Offers
net_profit = saved_revenue - wasted_spend

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC Score (Model Power)", f"{roc_auc_score(y_test, y_probs):.3f}")
col2.metric("Projected Net Profit", f"₹{net_profit:,.0f}", delta="Business KPI")
col3.metric("Customers Saved (TP)", f"{tp}")
col4.metric("Offers Sent (TP+FP)", f"{tp+fp}")

# --- TABS FOR DETAILED ANALYSIS ---
tab1, tab2, tab3, tab4 = st.tabs(["💰 Profit Analysis", "📉 Business Matrix", "🔍 Model Explainability", "🤖 GPT Interventions"])

with tab1:
    st.subheader("Profitability Curve: Finding the Sweet Spot")
    st.markdown("This chart explains **why** we chose our threshold. We simulate the profit for every possible threshold (0.0 to 1.0).")
    
    # Calculate profit for range
    t_range = np.arange(0.0, 1.01, 0.05)
    profits = []
    
    for t in t_range:
        p_temp = (y_probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, p_temp).ravel()
        # Profit = (Saved Users * Net Value) - (Wasted Users * Cost)
        p_val = (tp_t * (customer_value - coupon_cost)) - (fp_t * coupon_cost)
        profits.append(p_val)
    
    # Find Peak
    max_profit = max(profits)
    best_t = t_range[np.argmax(profits)]
    
    # Plotly Line Chart
    fig_profit = px.line(x=t_range, y=profits, labels={'x': 'Threshold Probability', 'y': 'Net Profit (₹)'},
                         title=f"Optimal Threshold is {best_t:.2f} (Max Profit: ₹{max_profit:,.0f})")
    
    # Add vertical line for current selection
    fig_profit.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current Selection")
    fig_profit.update_traces(line_color='#2ca02c', line_width=3)
    st.plotly_chart(fig_profit, use_container_width=True)
    
    st.info(f"**Insight:** At threshold **{best_t:.2f}**, we balance saving customers vs. wasting coupons. "
            f"If we go to 1.0, we send 0 offers and profit drops. If we go to 0.1, we spam everyone and profit drops.")

with tab2:
    st.subheader("Confusion Matrix with Business Context")
    
    # Prepare Data for Heatmap with Custom Text
    z = [[tn, fp], [fn, tp]]
    
    # Custom Text Labels for the squares
    annotations = [
        [f"Happy Customers<br>(TN: {tn})<br>Action: Do Nothing", f"Wasted Budget<br>(FP: {fp})<br>Cost: -₹{wasted_spend}"],
        [f"Lost Revenue<br>(FN: {fn})<br>Loss: -₹{fn*customer_value}", f"Saved Customers<br>(TP: {tp})<br>Value: +₹{saved_revenue}"]
    ]
    
    x_labels = ['Predicted: Stay', 'Predicted: Churn']
    y_labels = ['Actual: Stay', 'Actual: Churn']
    
    fig_cm = px.imshow(z, x=x_labels, y=y_labels, text_auto=True, color_continuous_scale='Blues', aspect="auto")
    
    # Update text inside the heatmap
    fig_cm.update_traces(text=annotations, texttemplate="%{text}")
    fig_cm.update_layout(title="Business Impact Matrix")
    
    st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.subheader("🔍 Model Explainability (Glass Box)")
    
    col_exp1, col_exp2 = st.columns(2)
    
    # --- 1. Logistic Regression (The "Why") ---
    with col_exp1:
        st.markdown("### 1. Logistic Regression (Direction)")
        st.caption("Green = Keeps Users. Red = Drives Churn.")
        
        # Extract Coefficients
        feature_names = (st.session_state['num_cols'] + 
                         list(log_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(st.session_state['cat_cols'])))
        coeffs = log_model.named_steps['classifier'].coef_[0]
        
        coeff_df = pd.DataFrame({'Feature': feature_names, 'Impact': coeffs})
        coeff_df = coeff_df.sort_values(by='Impact', ascending=False)
        
        # Color Logic
        coeff_df['Type'] = coeff_df['Impact'].apply(lambda x: 'Risk (Increases Churn)' if x > 0 else 'Retention (Reduces Churn)')
        
        fig_imp = px.bar(coeff_df, x='Impact', y='Feature', color='Type', orientation='h',
                         color_discrete_map={'Risk (Increases Churn)': '#d62728', 'Retention (Reduces Churn)': '#2ca02c'},
                         title="Linear Impact Factors")
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- 2. XGBoost (The "How Much") ---
    with col_exp2:
        st.markdown("### 2. XGBoost (Importance)")
        st.caption("Which features does the AI rely on the most?")
        
        # Extract Feature Importance
        xgb_clf = xgb_model.named_steps['classifier']
        importances = xgb_clf.feature_importances_
        
        xgb_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        xgb_df = xgb_df.sort_values(by='Importance', ascending=True) # Sort for chart
        
        fig_xgb = px.bar(xgb_df, x='Importance', y='Feature', orientation='h',
                         title="Feature Importance (Gain)",
                         color_discrete_sequence=['#1f77b4'])
        
        st.plotly_chart(fig_xgb, use_container_width=True)
        
    st.info("""
    **Interview Insight:**
    * **Logistic Regression** tells us *Direction*: e.g., "Being a Pro Member reduces churn."
    * **XGBoost** tells us *Intensity*: e.g., "Support Tickets is the #2 most important feature for prediction."
    """)

with tab4:
    st.subheader("🤖 GenAI Personalized Offers")
    st.markdown("Offers generated by the **Circuit Breaker Pipeline** (Phi-3.5 / Fallback).")
    
    # Load the GPT offers file
    gpt_path = "churn_prediction/data/top5_with_gpt_offers.csv"
    
    # Try multiple paths
    loaded_gpt = False
    possible_gpt_paths = [
        "churn_prediction/data/top5_with_gpt_offers.csv",
        "../data/top5_with_gpt_offers.csv",
        "C:/Users/HP/Desktop/zomato-genai-project/churn_prediction/data/top5_with_gpt_offers.csv"
    ]
    
    for p in possible_gpt_paths:
        try:
            gpt_df = pd.read_csv(p)
            st.success(f"Loaded offers from: {p}")
            # Display beautifully
            for i, row in gpt_df.iterrows():
                with st.expander(f"User {row['user_id']} (Risk: {row['prob_churn']:.1%})", expanded=True):
                    st.write(f"**💬 Message:** {row['gpt_offer']}")
            loaded_gpt = True
            break
        except:
            continue
            
    if not loaded_gpt:
        st.warning("⚠️ No GPT offers found. Please run 'hf_offer_generator.py' first.")