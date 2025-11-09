import os
import sqlite3
import pandas as pd
import torch
import datetime
import spacy
from tqdm import tqdm
from google_play_scraper import reviews, Sort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

def compute_hash(text: str, timestamp: str, source: str):
    key = f"{source}:{timestamp}:{text}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()

# ============================================================
# 0️⃣ Load ENV + OpenAI
# ============================================================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ai = OpenAI(api_key=OPENAI_KEY)

# ============================================================
# 1️⃣ Load Sentiment Model (RoBERTa)
# ============================================================
print("🔧 Loading RoBERTa sentiment model...")
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
LABELS = ["Negative", "Neutral", "Positive"]

def analyze_sentiment(text):
    text = text[:1000].strip()
    if not text:
        return "Neutral", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    i = probs.argmax()
    return LABELS[i], float(probs[i])

# ============================================================
# 2️⃣ Location Extraction (spaCy)
# ============================================================
nlp = spacy.load("en_core_web_sm")

def extract_location(text):
    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return locs[0] if locs else None

# ============================================================
# 3️⃣ GPT Topic + Severity Classifier
# ============================================================
def classify_metadata(text):
    prompt = f"""
    Classify this Play Store review about T-Mobile.

    Return JSON:
    {{
      "topic": "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | app | general",
      "severity": "low | medium | high"
    }}

    Review:
    \"\"\"{text}\"\"\"
    """
    try:
        r = ai.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=150
        )
        import json
        return json.loads(r.output_text)
    except:
        return {"topic": "general", "severity": "low"}

# ============================================================
# 4️⃣ Connect to SQLite Unified DB
# ============================================================
DB_PATH = "customer_happiness.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Make sure table exists
cur.execute("""
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    author TEXT,
    text TEXT,
    timestamp TEXT,
    url TEXT,
    location TEXT,
    score_raw REAL,
    sentiment_label TEXT,
    sentiment_score REAL,
    topic TEXT,
    severity TEXT
)
""")
conn.commit()

# ============================================================
# 5️⃣ Fetch Play Store Reviews
# ============================================================
print("📱 Fetching reviews for T-Mobile 'My T-Mobile' app...")

raw_reviews, _ = reviews(
    "com.tmobile.pr.mytmobile",
    lang="en",
    country="us",
    sort=Sort.NEWEST,
    count=200
)

print(f"📦 Collected {len(raw_reviews)} reviews.")

# ============================================================
# 6️⃣ Process, Classify, Insert into DB
# ============================================================
stored = 0

for r in tqdm(raw_reviews, desc="Processing reviews"):

    text = r.get("content", "").strip()
    if not text:
        continue

    timestamp = r["at"].isoformat()
    url = f"https://play.google.com/store/apps/details?id=com.tmobile.pr.mytmobile"

    # Sentiment
    sentiment, sentiment_score = analyze_sentiment(text)

    # Location (rarely exists but still parsed)
    location = extract_location(text)

    # GPT topic + severity
    meta = classify_metadata(text)

    cur.execute("""
    INSERT INTO insights (
        source, author, text, timestamp, url,
        location, score_raw,
        sentiment_label, sentiment_score,
        topic, severity
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "playstore",
        r.get("userName", "unknown"),
        text,
        timestamp,
        url,
        location,
        r.get("score", None),     # ⭐ star rating
        sentiment,
        sentiment_score,
        meta["topic"],
        meta["severity"]
    ))

    stored += 1

conn.commit()

print(f"\n💾 Stored {stored} Play Store rows into {DB_PATH}")

# ============================================================
# 7️⃣ Optional: Export snapshot
# ============================================================
df = pd.read_sql_query("SELECT * FROM insights WHERE source='playstore'", conn)
df.to_csv("playstore_snapshot.csv", index=False)
print("📊 Exported playstore_snapshot.csv")
