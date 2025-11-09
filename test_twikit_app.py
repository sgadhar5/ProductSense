import asyncio
import os
from twikit import Client
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# --- 0️⃣  Load environment variables ----------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

ai_client = OpenAI(api_key=api_key)

# --- 1️⃣  Classification Function -------------------------------------------
async def classify_tweets(tweets):
    """
    Use OpenAI to determine which tweets are truly about T-Mobile customer experience
    (network/service/price/support) with high precision.
    """
    relevant = []

    # Fast local pre-filter to skip obvious noise
    EXCLUDE_WORDS = {
        "t-mobile arena", "t mobile arena", "t-mobile center", "ticket", "tickets",
        "concert", "show", "vegas", "kansas city", "unlock", "imei", "sidekick",
        "credit card", "visa", "garage", "mariners", "giveaway", "promo code"
    }

    INCLUDE_HINTS = {
        "network", "signal", "coverage", "5g", "5 g", "lte", "internet",
        "home internet", "wifi", "wi-fi", "speed", "outage", "throttle",
        "hotspot", "plan", "billing", "customer service", "support", "csr",
        "call center", "activation", "port", "sim", "e-sim", "esim"
    }

    def likely_irrelevant(text: str) -> bool:
        t = text.lower()
        if any(w in t for w in EXCLUDE_WORDS):
            return True
        return False

    def post_filter(text: str) -> bool:
        t = text.lower()
        return not any(w in t for w in EXCLUDE_WORDS)

    SYSTEM_PROMPT = (
        "You are a strict relevance filter for brand customer-experience signals.\n"
        "Return EXACTLY one token: Relevant or Irrelevant.\n"
        "Relevant ONLY IF the tweet clearly concerns T-Mobile's:\n"
        " - network performance (signal, coverage, 5G/LTE, speed, outages, throttling, hotspot)\n"
        " - home internet / Wi-Fi experience\n"
        " - pricing/plans/billing (complaints, value, discounts)\n"
        " - customer service/support (store or online support interactions)\n"
        "Irrelevant examples: arenas/venues/concerts/tickets, celebrity gossip, credit cards/finance products,\n"
        "unlock/IMEI services, generic ads/promotions, politics not about service, jokes/memes without service content.\n"
        "Output MUST be exactly 'Relevant' or 'Irrelevant'."
    )

    FEWSHOT = [
        ('T-Mobile home internet keeps dropping every night. Support hasn’t fixed it.', 'Relevant'),
        ('Switched to T-Mobile 5G and speeds are way faster than my cable.', 'Relevant'),
        ('Autopay discount vanished on my T-Mobile bill. Support won’t help.', 'Relevant'),
        ('Selling 2 floor tickets for tonight at T-Mobile Arena, DM me.', 'Irrelevant'),
        ('. @tmobile just launched a credit card for rewards!', 'Irrelevant'),
        ('Unlock any T-Mobile/Verizon iPhone via IMEI fast service!', 'Irrelevant'),
        ('Fan cam from the concert at T-Mobile Center last night!', 'Irrelevant'),
    ]

    for t in tweets:
        text = t.text.replace("\n", " ").strip()
        if not text or likely_irrelevant(text):
            continue

        examples = "\n".join([f'Tweet: "{ex}"\nLabel: {lab}' for ex, lab in FEWSHOT])
        user_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{examples}\n\n"
            f'Tweet: "{text}"\nLabel:'
        )

        try:
            result = ai_client.responses.create(
                model="gpt-4.1-mini",
                input=user_prompt,
                max_output_tokens=32,
                temperature=0,
            )
            label = result.output_text.strip()
            if label == "Relevant" and post_filter(text):
                relevant.append(t)
        except Exception as e:
            print(f"⚠️ Error classifying tweet: {e}")

    return relevant

# --- 2️⃣  Main Function -----------------------------------------------------
async def main():
    client = Client("en-US")
    client.load_cookies("cookies_twikit.json")
    print("✅ Cookies loaded successfully!")

    me = await client.get_user_by_screen_name("hackutd2025")
    print(f"👤 Logged in as: {me.name} (@{me.screen_name})")

    # --- Pull up to 100 tweets via pagination --------------------------------
    query = '("T-Mobile") -arena -center -concert -tickets'
    print("\n🔍 Searching tweets...")
    tweets = await client.search_tweet(query, "Latest")

    max_pages = 5  # ≈ 20 tweets per page
    page = 1
    all_tweets = list(tweets)

    while page < max_pages:
        try:
            next_page = await tweets.next()
            if not next_page:
                break
            all_tweets.extend(next_page)
            page += 1
            print(f"📄 Loaded page {page}, total {len(all_tweets)} tweets...")
        except Exception as e:
            print(f"⚠️ Pagination stopped: {e}")
            break

    tweets = all_tweets
    print(f"📦 Pulled total {len(tweets)} tweets.\n")

    if not tweets:
        print("No tweets found — try again later.")
        return

    # --- 3️⃣  Filter with GPT -----------------------------------------------
    print("🤖 Filtering tweets with OpenAI (this may take a few minutes)...\n")
    relevant_tweets = await classify_tweets(tweets)

    print(f"✅ Found {len(relevant_tweets)} relevant tweets!\n")

    # --- 4️⃣  Display results ------------------------------------------------
    for i, t in enumerate(relevant_tweets, 1):
        print(f"{i}. {t.user.name}: {t.text}\n")

    # --- 5️⃣  Save to CSV ----------------------------------------------------
    df = pd.DataFrame([{"user": t.user.name, "text": t.text} for t in relevant_tweets])
    df.to_csv("relevant_tmobile_tweets.csv", index=False)
    print("💾 Saved relevant tweets to relevant_tmobile_tweets.csv")

# --- 3️⃣  Run ---------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
