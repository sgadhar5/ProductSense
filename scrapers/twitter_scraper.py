import asyncio
import os
import sqlite3
import datetime
import re
import torch
import spacy
from twikit import Client
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# 0️⃣ Load environment
# ---------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

ai_client = OpenAI(api_key=api_key)

# NLP for location extraction
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------
# 1️⃣ Load RoBERTa sentiment model
# ---------------------------------------------------------------------
print("🧠 Loading RoBERTa sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
labels = ['Negative', 'Neutral', 'Positive']

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return labels[idx], float(probs[idx])

# ---------------------------------------------------------------------
# 2️⃣ SQLite unified schema
# ---------------------------------------------------------------------
DB_PATH = "customer_happiness.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    author TEXT,
    text TEXT,
    timestamp TEXT,
    url TEXT,
    location TEXT,
    score_raw INTEGER,
    sentiment_label TEXT,
    sentiment_score REAL,
    topic TEXT,
    severity TEXT
)
""")
conn.commit()

# ---------------------------------------------------------------------
# 3️⃣ Utility — location extraction
# ---------------------------------------------------------------------
def extract_location(text, twitter_loc=None):
    # Use explicit Twitter user location if exists
    if twitter_loc and len(twitter_loc.strip()) > 2:
        return twitter_loc.strip()

    # NLP extraction from text
    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    if locs:
        return locs[0]

    # Regex fallback
    city_match = re.search(r"(in|at)\s+([A-Z][a-zA-Z]+)", text)
    if city_match:
        return city_match.group(2)

    return None

# ---------------------------------------------------------------------
# 4️⃣ GPT metadata classifier (topic + severity)
# ---------------------------------------------------------------------
def classify_metadata(text):
    prompt = f"""
You label customer experience messages for telecom companies.

Return JSON only:
{{
  "topic": "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | general",
  "severity": "low | medium | high"
}}

Classify this message:
\"\"\"{text}\"\"\"
"""

    try:
        r = ai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=120,
            temperature=0
        )
        structured = r.output_text.strip()
        import json
        return json.loads(structured)
    except:
        return {"topic": "general", "severity": "low"}

# ---------------------------------------------------------------------
# 5️⃣ GPT relevance filter
# ---------------------------------------------------------------------
async def classify_relevance(tweets):
    filtered = []

    SYSTEM_PROMPT = (
        "You are a strict classifier. Output ONLY: Relevant or Irrelevant.\n"
        "Relevant ONLY if the tweet is about T-Mobile customer experience: "
        "billing, pricing, outages, 5G, support, customer service, device trade-ins."
    )

    examples = """
Tweet: "T-Mobile internet keeps dropping every night." → Relevant
Tweet: "Selling tickets at T-Mobile Arena." → Irrelevant
"""

    for t in tweets:
        text = t.text.replace("\n", " ").strip()
        if not text:
            continue

        prompt = f"{SYSTEM_PROMPT}\n\n{examples}\nTweet: \"{text}\"\nLabel:"

        try:
            r = ai_client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                temperature=0,
                max_output_tokens=5
            )
            label = r.output_text.strip()
            if label == "Relevant":
                filtered.append(t)
        except Exception as e:
            print("⚠️ GPT filtering error:", e)

    return filtered

# ---------------------------------------------------------------------
# 6️⃣ MAIN: ingest → classify → store in DB
# ---------------------------------------------------------------------
async def main():
    client = Client("en-US")
    client.load_cookies("cookies_twikit.json")
    print("🍪 Twitter cookies loaded.")

    me = await client.get_user_by_screen_name("hackutd2025")
    print(f"👤 Logged in as {me.screen_name}")

    # ------------ Pull tweets ------------
    query = '("T-Mobile") -arena -concert -tickets'
    print("🔍 Searching tweets…")
    tweets = await client.search_tweet(query, "Latest")

    all_tweets = list(tweets)
    for _ in range(5):
        try:
            next_pg = await tweets.next()
            if not next_pg:
                break
            all_tweets.extend(next_pg)
        except:
            break

    print(f"📦 Pulled {len(all_tweets)} tweets.")

    # ------------ Filter relevance ------------
    print("🤖 Filtering for relevant tweets with GPT…")
    relevant = await classify_relevance(all_tweets)
    print(f"🎯 {len(relevant)} relevant tweets found.")

    # ------------ Process each tweet ------------
    for t in relevant:
        text = t.text.strip()
        ts = getattr(t, "created_at", datetime.datetime.utcnow().isoformat())
        sentiment_label, sentiment_score = get_sentiment(text)

        # Extract metadata
        location = extract_location(text, twitter_loc=t.user.location)
        meta = classify_metadata(text)

        # Insert into DB
        cur.execute("""
        INSERT INTO insights (
            source, author, text, timestamp, url, location, score_raw,
            sentiment_label, sentiment_score, topic, severity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "twitter",
            t.user.name,
            text,
            ts,
            f"https://twitter.com/{t.user.screen_name}/status/{t.id}",
            location,
            t.favorite_count if hasattr(t, "favorite_count") else None,
            sentiment_label,
            sentiment_score,
            meta["topic"],
            meta["severity"]
        ))

    conn.commit()
    print("💾 Stored into DB successfully.")

    df = pd.read_sql_query("SELECT * FROM insights WHERE source='twitter'", conn)
    df.to_csv("twitter_snapshot.csv", index=False)
    print("📊 Exported twitter_snapshot.csv")

# Run
if __name__ == "__main__":
    asyncio.run(main())
