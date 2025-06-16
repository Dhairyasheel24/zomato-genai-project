# 🍽️ Zomato GenAI – Churn Prediction & Retention Strategy

## 👤 Use Case: Zomato CX Manager  
Many users silently stop ordering without formally leaving. Identifying such users and proactively reaching out with personalized offers can significantly reduce churn and boost revenue.

---

## 🎯 Objective  
Use Machine Learning and Generative AI (GPT) to:
- 🔍 Predict customers most likely to churn  
- 📩 Generate personalized retention messages  
- 📊 Display everything on an interactive Streamlit dashboard  

---

## 🧠 AI-Powered Workflow

| Module             | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| 🔍 ML Model        | Churn prediction using Logistic Regression / RandomForest / XGBoost |
| 🤖 GPT Integration | Generates friendly 2-line messages with discounts and urgency       |
| 📊 Streamlit UI    | Interactive dashboard to explore churn and AI-generated insights    |

---

## 💡 Business Impact
- 🔻 10–20% reduction in user churn  
- 📈 +5% improvement in monthly retention  
- ✉️ 100% automated messaging for re-engagement  

---

## 🛠️ Tech Stack
- **Languages/ML:** Python, Pandas, Scikit-learn, XGBoost  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **GenAI:** Hugging Face Inference API (Free GPT model)  
- **Frontend:** Streamlit  
- **Deployment:** GitHub + Streamlit Cloud  

---

## 📊 Key Metrics (KPIs)
- ✅ ROC AUC Score for churn prediction  
- 📩 GPT offer generation coverage for top churn-risk users  
- 🔁 Repeat order / redemption rate (future tracking)  

---

## 🚀 Live Demo  
🔗 [Zomato GenAI Streamlit App](https://dhairyasheel24-zomato-genai-churn-predictiondashboardapp-a6cphk.streamlit.app/)

---

## 🗂️ Project Structure (Partial)
zomato-genai-project/
│
├── churn_prediction/ # Main project logic
│ ├── data/ # Input datasets and prediction outputs
│ │ ├── zomato_churn_synthetic.csv
│ │ ├── top5_churn_risk_users.csv
│ │ └── top5_with_gpt_offers.csv
│ │
│ ├── dashboard/ # Streamlit dashboard application
│ │ └── app.py
│ │
│ ├── apps/ # GPT-powered offer generator logic
│ │ └── hf_offer_generator.py
│ │
│ ├── assets/ # Visuals and brand assets
│ └── Zomato_logo.png
│
├── notebooks/ # Jupyter notebooks (if any)
├── reports/ # Optional evaluation results
├── README.md # Project overview and instructions
├── requirements.txt # Python dependencies
└── .devcontainer/ # Optional dev container setup
