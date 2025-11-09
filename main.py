import os
import json
import sqlite3
import datetime
import asyncio
from typing import List, Optional, Literal, Dict, Any

# Silence HF tokenizers fork warning (optional)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Third-party clients/models ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# Reddit
import praw

# Twitter (Twikit: cookie-based)
from twikit import Client as TwClient

# Play Store
from google_play_scraper import reviews, Sort

# ============================================================
# 0) Config & App
# ============================================================
load_dotenv()
app = FastAPI(title="Customer Happiness Ingestion API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in env")
oaiclient = OpenAI(api_key=OPENAI_API_KEY)

# Reddit credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "tmobile_sentiment_app")

# --- TWITTER: use EXACT working cookie file + username (like your script) ---
TWITTER_COOKIES_PATH = os.getenv("TWITTER_COOKIES_PATH", "cookies_twikit.json")
TWITTER_WORKING_USERNAME = os.getenv("TWITTER_WORKING_USERNAME", "hackutd2025")

DB_PATH = os.getenv("DB_PATH", "customer_happiness.db")

# ============================================================
# 1) DB bootstrap
# ============================================================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
    return conn

# ============================================================
# 2) Models (Pydantic)
# ============================================================
SourceType = Literal["twitter","reddit_post","reddit_comment","playstore"]

class IngestResult(BaseModel):
    source: SourceType
    inserted: int
    details: Optional[Dict[str, Any]] = None

class Insight(BaseModel):
    source: SourceType
    author: Optional[str]
    text: str
    timestamp: str
    url: Optional[str]
    location: Optional[str]
    score_raw: Optional[float]
    sentiment_label: Literal["Negative","Neutral","Positive"]
    sentiment_score: float
    topic: Literal["billing","5G","outage","pricing","customer_service","devices","trade-ins","autopay","app","general"]
    severity: Literal["low","medium","high"]

# ============================================================
# 3) Sentiment model (RoBERTa)
# ============================================================
print("🧠 Loading RoBERTa sentiment model...")
_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device).eval()
_LABELS = ["Negative", "Neutral", "Positive"]

def analyze_sentiment(text: str):
    text = (text or "").strip()
    if not text:
        return "Neutral", 0.0
    text = text[:1000]
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    i = int(probs.argmax())
    return _LABELS[i], float(probs[i])

# ============================================================
# 4) Topic + Severity via GPT
# ============================================================
TOPIC_ENUM = "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | app | general"

def classify_topic_severity(text: str) -> Dict[str, str]:
    prompt = f"""
Return ONLY JSON.

Classify this customer text into:
- "topic": one of [{TOPIC_ENUM}]
- "severity": "low" | "medium" | "high"

Text:
\"\"\"{text[:1200]}\"\"\"
"""
    try:
        r = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=120
        )
        return json.loads(r.output_text)
    except Exception:
        return {"topic": "general", "severity": "low"}

# ============================================================
# 5) GPT Location Extraction
# ============================================================
def extract_location(text: str) -> Optional[str]:
    if not text or len(text.strip()) == 0:
        return None
    prompt = f"""
Return ONLY JSON in the format:
{{ "location": <string or null> }}

Extract a geographic location (city, state, region, or country)
from the following message. 
If no location is mentioned, return null.

Message:
\"\"\"{text[:800]}\"\"\"
"""
    try:
        r = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=60
        )
        data = json.loads(r.output_text)
        return data.get("location") or None
    except Exception:
        return None

# ============================================================
# 6) Twitter (Twikit) — EXACT WORKING FLOW
#     Define BEFORE any routes that call it.
# ============================================================
async def twitter_client() -> TwClient:
    client = TwClient("en-US")
    # Load EXACT working cookie file (no conversion)
    try:
        client.load_cookies(TWITTER_COOKIES_PATH)
        print(f"🔥 Loaded Twikit cookies from {TWITTER_COOKIES_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load Twikit cookies: {e}")

    # Same login check you used in the working script
    try:
        me = await client.get_user_by_screen_name(TWITTER_WORKING_USERNAME)
        print(f"🔐 Logged in as: {me.name} (@{me.screen_name})")
    except Exception as e:
        raise RuntimeError(f"Twikit cookie session invalid: {e}")

    return client

async def ingest_twitter(max_pages=5) -> IngestResult:
    client = await twitter_client()

    query = '("T-Mobile") -arena -center -concert -tickets'
    print("\n🔍 Searching tweets...")
    tweets = await client.search_tweet(query, "Latest")

    all_tweets = list(tweets)
    page = 1
    while page < max_pages:
        try:
            nxt = await tweets.next()
            if not nxt:
                break
            all_tweets.extend(nxt)
            page += 1
            print(f"📄 Loaded page {page}, total {len(all_tweets)} tweets...")
        except Exception as e:
            print(f"⚠️ Pagination stopped: {e}")
            break

    print(f"📦 Pulled total {len(all_tweets)} tweets.\n")

    rows: List[Insight] = []
    for t in all_tweets:
        text = (t.text or "").strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        ts = getattr(t, "created_at", None)
        ts_iso = ts if isinstance(ts, str) else datetime.datetime.utcnow().isoformat()
        url = f"https://x.com/{t.user.screen_name}/status/{t.id}"

        ins = Insight(
            source="twitter",
            author=getattr(t.user, "name", None),
            text=text[:2000],
            timestamp=ts_iso,
            url=url,
            location=extract_location(text),
            score_raw=float(getattr(t, "favorite_count", 0) or 0) +
                      float(getattr(t, "retweet_count", 0) or 0),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"],
        )
        rows.append(ins)

    conn = db()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT INTO insights (source, author, text, timestamp, url, location, score_raw,
                                  sentiment_label, sentiment_score, topic, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location, r.score_raw,
            r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()
    print(f"💾 Inserted {len(rows)} tweets into DB.")
    return IngestResult(source="twitter", inserted=len(rows))

# ============================================================
# 7) Source: Reddit
# ============================================================
def reddit_client():
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        raise RuntimeError("Reddit credentials missing in env")
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def ingest_reddit(limit_search=50, limit_comments=200) -> IngestResult:
    reddit = reddit_client()
    rows: List[Insight] = []

    # Search site-wide
    for sub in reddit.subreddit("all").search("T-Mobile OR TMobile", limit=limit_search):
        text = f"{sub.title or ''} {sub.selftext or ''}".strip()
        if not text:
            continue
        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)
        ins = Insight(
            source="reddit_post",
            author=str(sub.author) if sub.author else None,
            text=text[:2000],
            timestamp=datetime.datetime.utcfromtimestamp(sub.created_utc).isoformat(),
            url=sub.url,
            location=extract_location(text),
            score_raw=float(getattr(sub, "score", 0)),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"],
        )
        rows.append(ins)

    # Latest comments in r/TMobile
    for c in reddit.subreddit("TMobile").comments(limit=limit_comments):
        text = (c.body or "").strip()
        if not text:
            continue
        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)
        ins = Insight(
            source="reddit_comment",
            author=str(c.author) if c.author else None,
            text=text[:2000],
            timestamp=datetime.datetime.utcfromtimestamp(c.created_utc).isoformat(),
            url=f"https://reddit.com{c.permalink}",
            location=extract_location(text),
            score_raw=float(getattr(c, "score", 0)),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"],
        )
        rows.append(ins)

    conn = db()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT INTO insights (source, author, text, timestamp, url, location, score_raw,
                                  sentiment_label, sentiment_score, topic, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location, r.score_raw,
            r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()
    return IngestResult(source="reddit_post", inserted=len(rows))

# ============================================================
# 8) Source: Play Store
# ============================================================
def ingest_playstore(count=200) -> IngestResult:
    raw, _ = reviews(
        "com.tmobile.pr.mytmobile",
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=count
    )
    rows: List[Insight] = []
    for r in raw:
        text = (r.get("content") or "").strip()
        if not text:
            continue
        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)
        timestamp = r["at"].isoformat() if r.get("at") else datetime.datetime.utcnow().isoformat()
        url = "https://play.google.com/store/apps/details?id=com.tmobile.pr.mytmobile"

        ins = Insight(
            source="playstore",
            author=r.get("userName") or None,
            text=text[:2000],
            timestamp=timestamp,
            url=url,
            location=extract_location(text),
            score_raw=float(r.get("score") or 0),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"],
        )
        rows.append(ins)

    conn = db()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT INTO insights (source, author, text, timestamp, url, location, score_raw,
                                  sentiment_label, sentiment_score, topic, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location, r.score_raw,
            r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()
    return IngestResult(source="playstore", inserted=len(rows))

# ============================================================
# 9) API Routes
# ============================================================
@app.get("/health")
def health():
    return {"ok": True, "db": os.path.abspath(DB_PATH)}

@app.post("/ingest/reddit", response_model=IngestResult)
def api_ingest_reddit(limit_search: int = 50, limit_comments: int = 200):
    try:
        return ingest_reddit(limit_search=limit_search, limit_comments=limit_comments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/playstore", response_model=IngestResult)
def api_ingest_playstore(count: int = 200):
    try:
        return ingest_playstore(count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/twitter", response_model=IngestResult)
async def api_ingest_twitter(max_pages: int = 5):
    try:
        return await ingest_twitter(max_pages=max_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/all")
async def api_ingest_all():
    results = []
    try:
        t_task = asyncio.create_task(ingest_twitter(max_pages=5))
        loop = asyncio.get_event_loop()
        r_res = await loop.run_in_executor(None, ingest_reddit, 50, 200)
        p_res = await loop.run_in_executor(None, ingest_playstore, 200)
        t_res = await t_task
        results.extend([r_res.model_dump(), p_res.model_dump(), t_res.model_dump()])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ingested": results}

@app.get("/insights/recent", response_model=List[Insight])
def api_recent(limit: int = 50):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT source, author, text, timestamp, url, location, score_raw,
               sentiment_label, sentiment_score, topic, severity
        FROM insights
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        Insight(
            source=row[0], author=row[1], text=row[2], timestamp=row[3], url=row[4],
            location=row[5], score_raw=row[6],
            sentiment_label=row[7], sentiment_score=row[8],
            topic=row[9], severity=row[10]
        )
        for row in rows
    ]

@app.get("/insights/summary")
def api_summary():
    conn = db()
    cur = conn.cursor()
    cur.execute("""SELECT sentiment_label, COUNT(*) FROM insights GROUP BY sentiment_label""")
    sents = dict(cur.fetchall())
    cur.execute("""SELECT topic, COUNT(*) FROM insights GROUP BY topic""")
    topics = dict(cur.fetchall())
    cur.execute("""SELECT severity, COUNT(*) FROM insights GROUP BY severity""")
    sevs = dict(cur.fetchall())
    conn.close()
    return {"sentiment_counts": sents, "topic_counts": topics, "severity_counts": sevs}
