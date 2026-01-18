# 🍽️ Zomato GenAI – Churn Prediction & Retention System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![GenAI](https://img.shields.io/badge/GenAI-HuggingFace-yellow)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

> **"Prediction is just the first step. Prevention is the goal."**

### 👤 Use Case: Zomato Customer Experience (CX) Manager
Many users silently stop ordering without formally cancelling their accounts. Identifying these "silent churners" and proactively reaching out with **personalized, profitable offers** can significantly reduce churn and boost Customer Lifetime Value (LTV).

---

## 🎯 Objective
This project is an **End-to-End Retention Pipeline** that goes beyond simple prediction. It uses Machine Learning to identify risk and Generative AI to automate intervention.

1.  **🔍 Predict:** Identify customers most likely to churn using **XGBoost**.
2.  **💰 Optimize:** Dynamically tune decision thresholds based on **Net Profit** (LTV vs. Cost), not just accuracy.
3.  **📩 Intervene:** Generate personalized push notifications using **LLMs (Phi-3.5/Qwen)**.
4.  **🛡️ Resilient:** Implements a **Circuit Breaker Pattern** to ensure marketing pipelines never fail even if AI APIs go down.

---

## 🧠 Architecture & Workflow

| Module | Technology | Role |
| :--- | :--- | :--- |
| **Prediction Engine** | **XGBoost Classifier** | High-performance churn probability scoring on imbalanced data. |
| **Explainability** | **Logistic Regression** | Provides "Directional" insights (e.g., *Why* are they leaving?). |
| **GenAI Agent** | **Hugging Face API** | Generates context-aware retention offers (e.g., "We miss you! 50% off"). |
| **Dashboard** | **Streamlit** | Interactive UI for stakeholders to simulate profit scenarios. |

---

## 💡 Key Features (The "Why" Behind the Code)

### 1. 💰 Profit-First Optimization
Most models optimize for Accuracy. This project optimizes for **Money**.
* We calculate the trade-off between **saving a customer (LTV ₹500)** and **wasting a coupon (Cost ₹50)**.
* The dashboard plots a **Profitability Curve** to find the optimal decision threshold (usually ~0.35 instead of the default 0.5).

### 2. 🛡️ GenAI Circuit Breaker
Reliance on external APIs (like OpenAI/Hugging Face) is risky in production.
* **Primary:** Calls the live LLM (Microsoft Phi-3.5) for fresh content.
* **Fallback:** If the API times out or fails (Error 503), the system automatically switches to a deterministic **"Backup Library"** of high-converting offers.
* **Result:** 100% Uptime Guarantee for the marketing pipeline.

### 3. 📉 Business Confusion Matrix
We redefined standard ML metrics into business terms for stakeholders:
* **True Positive:** ✅ "Saved Customer" (Revenue Gained)
* **False Positive:** 💸 "Wasted Budget" (Coupon Cost Lost)
* **False Negative:** 📉 "Lost Revenue" (The most expensive mistake)

---

## 🛠️ Tech Stack

* **Core:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost (with `scale_pos_weight` for class imbalance)
* **Generative AI:** Hugging Face Inference API (Phi-3.5, Qwen 2.5)
* **Visualization:** Plotly (Interactive Charts), Matplotlib
* **Web App:** Streamlit

---

## 📊 Project Structure

```bash
├── churn_prediction/
│   ├── data/                       # Synthetic datasets & generated offers
│   │   ├── zomato_churn_synthetic.csv
│   │   ├── top5_churn_risk_users.csv
│   │   └── top5_with_gpt_offers.csv
│   │
│   ├── dashboard/                  # Interactive Streamlit App
│   │   └── app.py                  # MAIN DASHBOARD SCRIPT
│   │
│   ├── apps/                       # Background Jobs
│   │   └── hf_offer_generator.py   # GenAI Script (with Circuit Breaker)
│   │
│   └── models/                     # (Optional) Saved .pkl models
│
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
