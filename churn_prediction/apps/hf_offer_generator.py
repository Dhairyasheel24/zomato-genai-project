import requests
import pandas as pd
import time

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {
    "Authorization": "Bearer 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"
}

def query(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 60,
            "temperature": 0.7,
            "return_full_text": False  # 🧠 Important: only return generation
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            return "⚠️ Unexpected response format."
        except Exception as e:
            return f"⚠️ JSON parse error: {e}"
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

def main():
    try:
        df = pd.read_csv("C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\data\\top5_churn_risk_users.csv")
    except FileNotFoundError:
        print("⚠️ CSV file not found.")
        return

    print("🔮 Generating personalized retention offers...\n")
    offers = []

    for _, row in df.iterrows():
        prompt = f"""Hey {row['user_id']}, we noticed you haven't ordered in a while!
Enjoy 20% off your next Zomato order – just for today!"""

        gpt_response = query(prompt)

        print(f"👤 {row['user_id']} ({row['prob_churn']:.2%})")
        print(f"💬 {gpt_response}\n")

        offers.append({
            "user_id": row["user_id"],
            "prob_churn": row["prob_churn"],
            "gpt_offer": gpt_response
        })

        time.sleep(1)

    output_path = "C:\\Users\\HP\\Desktop\\zomato-genai-project\\churn_prediction\\data\\top5_with_gpt_offers.csv"
    pd.DataFrame(offers).to_csv(output_path, index=False)
    print(f"\n✅ Offers saved to: {output_path}")

if __name__ == "__main__":
    main()
