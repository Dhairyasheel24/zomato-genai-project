import requests
import pandas as pd
import time
import random
import os

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")

# Primary Model (Microsoft Phi-3.5 is usually most stable)
API_URL = "https://router.huggingface.co/hf-inference/models/microsoft/Phi-3.5-mini-instruct"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# --- BACKUP LIBRARY (Circuit Breaker) ---
# If the API fails, we randomly pick one of these to ensure the demo never breaks.
BACKUP_OFFERS = [
    "We miss you! Flat 50% OFF your next order. Order Now!",
    "Hungry? Your favorites are waiting. Get FREE delivery today.",
    "Long time no see! Here's a ₹100 coupon just for you.",
    "Don't cook tonight! Enjoy 30% OFF on top rated restaurants.",
    "Craving Biryani? Order now and get a complimentary dessert!",
    "Weekend Special: Buy 1 Get 1 Free on select items!",
    "Your tastebuds miss you. 40% OFF valid for next 2 hours.",
    "Mid-week blues? Cheer up with ₹150 OFF your dinner.",
    "Flash Sale! Get 25% cashback on your next order.",
    "Treat yourself! Free delivery on orders above ₹199."
]

def query_resilient(prompt):
    """
    Attempts to use Real AI. If it fails/timeouts, uses High-Quality Mock Data.
    """
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        # Try calling the API with a short timeout (5 seconds)
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                text = result[0]["generated_text"].strip()
                # Basic validation to ensure it didn't return garbage
                if len(text) > 5:
                    return text, "✅ AI Generated"
                    
    except Exception as e:
        pass # Silently catch errors to trigger fallback

    # FALLBACK MODE
    # If we reach here, the API failed. Return a backup offer.
    return random.choice(BACKUP_OFFERS), "⚠️ Backup (API Down)"

def main():
    # 1. Load Data
    paths = [
        "churn_prediction/data/top5_churn_risk_users.csv",
        "../data/top5_churn_risk_users.csv",
        "C:/Users/HP/Desktop/zomato-genai-project/churn_prediction/data/top5_churn_risk_users.csv"
    ]
    
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            print(f"✅ Loaded input data from: {p}")
            break
        except:
            continue
            
    if df is None:
        print("❌ Error: Input CSV not found. Run 'churn_model.py' first.")
        return

    print("🔮 Generating offers (Circuit Breaker Mode Enabled)...")
    print("-----------------------------------------------------")
    
    offers = []

    for _, row in df.iterrows():
        # Create prompt
        prompt = (
            f"User {row['user_id']} is at {row['prob_churn']:.0%} risk of churning from Zomato. "
            f"Write a 10-word push notification offering a discount."
        )

        # Get response (either Real AI or Backup)
        offer_text, source = query_resilient(prompt)
        
        print(f"👤 {row['user_id']}: {offer_text} [{source}]")

        offers.append({
            "user_id": row["user_id"],
            "prob_churn": row["prob_churn"],
            "gpt_offer": offer_text.replace('"', '')
        })
        
        # Short sleep to mimic processing time
        time.sleep(1)

    # 2. Save Data
    # We force the save to the specific path your dashboard looks for
    output_path = "churn_prediction/data/top5_with_gpt_offers.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        pd.DataFrame(offers).to_csv(output_path, index=False)
        print(f"\n✅ Success! Offers saved to: {output_path}")
        print("   (You can now run 'streamlit run app.py')")
    except Exception as e:
        # Fallback to absolute path if relative fails
        abs_path = "C:/Users/HP/Desktop/zomato-genai-project/churn_prediction/data/top5_with_gpt_offers.csv"
        pd.DataFrame(offers).to_csv(abs_path, index=False)
        print(f"\n✅ Success! Saved to absolute path: {abs_path}")

if __name__ == "__main__":
    main()