import os
import sqlite3
import pandas as pd
import torch
import datetime
import spacy
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import praw
from openai import OpenAI
import hashlib

def compute_hash(text: str, timestamp: str, source: str):
    key = f"{source}:{timestamp}:{text}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()

# ============================================================
# 0️⃣ Load ENV + OpenAI
# ============================================================
load_dotenv()
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "tmobile_sentiment_app")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("❌ Missing Reddit API credentials")

ai = OpenAI(api_key=OPENAI_KEY)

# ============================================================
# 1️⃣ Init Reddit API
# ============================================================
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ============================================================
# 2️⃣ Load Sentiment Model (RoBERTa)
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    i = probs.argmax()
    return LABELS[i], float(probs[i])

# ============================================================
# 3️⃣ NLP Location Extraction
# ============================================================
nlp = spacy.load("en_core_web_sm")

def extract_location(text):
    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return locs[0] if locs else None

# ============================================================
# 4️⃣ GPT Topic + Severity Classifier
# ============================================================
def classify_metadata(text):
    prompt = f"""
    Classify the T-Mobile related message.

    Return JSON:
    {{
      "topic": "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | general",
      "severity": "low | medium | high"
    }}

    Message:
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
# 5️⃣ Connect to unified DB
# ============================================================
DB_PATH = "customer_happiness.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ============================================================
# 6️⃣ Pull Reddit Posts and Comments
# ============================================================
query = "T-Mobile OR TMobile"
print(f"🔍 Searching Reddit for '{query}'...\n")

posts = []

# ---- Posts mentioning T-Mobile globally ----
for submission in reddit.subreddit("all").search(query, limit=50):
    posts.append({
        "source": "reddit_post",
        "author": str(submission.author),
        "text": (submission.title + " " + submission.selftext).strip(),
        "url": f"https://reddit.com{submission.permalink}",
        "timestamp": datetime.datetime.utcfromtimestamp(submission.created_utc).isoformat(),
        "score_raw": submission.score
    })

# ---- Comments inside r/TMobile ----
print("💬 Pulling comments from r/TMobile...\n")
for c in reddit.subreddit("TMobile").comments(limit=200):
    posts.append({
        "source": "reddit_comment",
        "author": str(c.author),
        "text": c.body.strip(),
        "url": f"https://reddit.com{c.permalink}",
        "timestamp": datetime.datetime.utcfromtimestamp(c.created_utc).isoformat(),
        "score_raw": c.score
    })

print(f"📦 Collected {len(posts)} raw Reddit items.\n")

# ============================================================
# 7️⃣ Process → classify → store
# ============================================================
stored = 0

for p in posts:
    text = p["text"]
    if not text:
        continue

    # Sentiment
    sentiment, sentiment_score = analyze_sentiment(text)

    # Location (optional)
    location = extract_location(text)

    # Metadata classification
    meta = classify_metadata(text)

    # Insert into database
    cur.execute("""
    INSERT INTO insights (
        source, author, text, timestamp, url,
        location, score_raw,
        sentiment_label, sentiment_score,
        topic, severity
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        p["source"],
        p["author"],
        text,
        p["timestamp"],
        p["url"],
        location,
        p["score_raw"],
        sentiment,
        sentiment_score,
        meta["topic"],
        meta["severity"]
    ))

    stored += 1

conn.commit()
print(f"💾 Stored {stored} Reddit rows into {DB_PATH}")

# ============================================================
# 8️⃣ Optional CSV Export
# ============================================================
df = pd.read_sql_query("SELECT * FROM insights WHERE source LIKE 'reddit%'", conn)
df.to_csv("reddit_snapshot.csv", index=False)
print("📊 Exported reddit_snapshot.csv")
