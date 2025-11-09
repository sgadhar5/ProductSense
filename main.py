import os
import json
import sqlite3
import datetime
import asyncio
from typing import List, Optional, Literal, Dict, Any

# Silence HF tokenizers fork warning (optional)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ML + API Clients
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# Reddit
import praw

# Twitter (Twikit)
from twikit import Client as TwClient

# Play Store
from google_play_scraper import reviews, Sort


# ============================================================
# 0) CONFIG
# ============================================================
load_dotenv()

app = FastAPI(title="Customer Happiness Ingestion API", version="2.1.0")

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

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "tmobile_dashboard_app")

# Use the exact cookie file + username that work in your standalone test
TWITTER_COOKIES_PATH = os.getenv("TWITTER_COOKIES_PATH", "cookies_twikit.json")
TWITTER_WORKING_USERNAME = os.getenv("TWITTER_WORKING_USERNAME", "hackutd2025")

DB_PATH = os.getenv("DB_PATH", "customer_happiness.db")


# ============================================================
# 1) DATABASE + CHECKPOINTS
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
        url TEXT UNIQUE,
        location TEXT,
        score_raw REAL,
        sentiment_label TEXT,
        sentiment_score REAL,
        topic TEXT,
        severity TEXT
    );
    """)
    # Checkpoints:
    # - twitter: last tweet id (stringified int)
    # - reddit: last UNIX timestamp (float as string)
    # - playstore: last UNIX timestamp (float as string)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS checkpoints (
        source TEXT PRIMARY KEY,
        last_value TEXT
    );
    """)
    conn.commit()
    return conn


def get_checkpoint(source: str) -> Optional[str]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT last_value FROM checkpoints WHERE source=?", (source,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_checkpoint(source: str, value: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO checkpoints (source, last_value)
        VALUES (?, ?)
        ON CONFLICT(source) DO UPDATE SET last_value=excluded.last_value
    """, (source, value))
    conn.commit()
    conn.close()


# ============================================================
# 2) MODELS (Pydantic)
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
# 3) SENTIMENT (RoBERTa)
# ============================================================
print("🔼 Loading RoBERTa sentiment model...")
_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device).eval()
_LABELS = ["Negative", "Neutral", "Positive"]


def analyze_sentiment(text: str):
    text = (text or "").strip()[:1000]
    if not text:
        return "Neutral", 0.0
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return _LABELS[idx], float(probs[idx])


# ============================================================
# 4) GPT TOPIC + SEVERITY
# ============================================================
TOPIC_ENUM = "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | app | general"

def classify_topic_severity(text: str) -> Dict[str, str]:
    prompt = f"""
Return ONLY JSON.
Classify:
- "topic": one of [{TOPIC_ENUM}]
- "severity": "low" | "medium" | "high"

Text:
\"\"\"{(text or '')[:1200]}\"\"\"
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
# 5) GPT LOCATION EXTRACTION
# ============================================================
def extract_location(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    prompt = f"""
Return ONLY JSON as: {{"location": <string or null>}}
Extract a geographic location (city/state/region/country) if explicitly mentioned.

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
        loc = data.get("location")
        return loc if (isinstance(loc, str) and loc.strip()) else None
    except Exception:
        return None


# ============================================================
# 6) TWITTER (DELTA MODE + EXACT WORKING LOGIN)
# ============================================================
async def twitter_client() -> TwClient:
    client = TwClient("en-US")
    try:
        client.load_cookies(TWITTER_COOKIES_PATH)
        print(f"✅ Loaded Twitter cookies from {TWITTER_COOKIES_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load Twikit cookies: {e}")

    try:
        me = await client.get_user_by_screen_name(TWITTER_WORKING_USERNAME)
        print(f"🔐 Logged in as {me.screen_name}")
    except Exception as e:
        raise RuntimeError(f"Invalid Twikit cookies/session: {e}")

    return client


async def ingest_twitter(max_pages=5) -> IngestResult:
    client = await twitter_client()

    query = '("T-Mobile") -arena -center -concert -tickets'
    last_seen_id_raw = get_checkpoint("twitter")
    last_seen_id = int(last_seen_id_raw) if (last_seen_id_raw and last_seen_id_raw.isdigit()) else 0
    print(f"📌 Twitter checkpoint (last tweet id): {last_seen_id}")

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
        except Exception:
            break

    # Delta: only newer than last_seen_id
    new_tweets = []
    max_id = last_seen_id
    for t in all_tweets:
        try:
            tid = int(getattr(t, "id"))
        except Exception:
            continue
        if tid <= last_seen_id:
            continue
        new_tweets.append(t)
        if tid > max_id:
            max_id = tid

    if not new_tweets:
        return IngestResult(source="twitter", inserted=0)

    rows: List[Insight] = []
    for t in new_tweets:
        text = (t.text or "").strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        # created_at can be datetime or string; normalize to ISO
        ts = getattr(t, "created_at", None)
        if ts is None:
            ts_iso = datetime.datetime.utcnow().isoformat()
        elif isinstance(ts, str):
            ts_iso = ts
        else:
            # assume datetime-like
            try:
                ts_iso = ts.isoformat()
            except Exception:
                ts_iso = datetime.datetime.utcnow().isoformat()

        url = f"https://x.com/{t.user.screen_name}/status/{t.id}"
        score_raw = float(getattr(t, "favorite_count", 0) or 0) + float(getattr(t, "retweet_count", 0) or 0)

        rows.append(Insight(
            source="twitter",
            author=getattr(t.user, "name", None),
            text=text[:2000],
            timestamp=ts_iso,
            url=url,
            location=extract_location(text),
            score_raw=score_raw,
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"]
        ))

    conn = db()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT OR IGNORE INTO insights (
                source, author, text, timestamp, url, location,
                score_raw, sentiment_label, sentiment_score, topic, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location,
            r.score_raw, r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()

    if max_id > last_seen_id:
        set_checkpoint("twitter", str(max_id))

    return IngestResult(source="twitter", inserted=len(rows))


# ============================================================
# 7) REDDIT (DELTA MODE)
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

    last_ts_raw = get_checkpoint("reddit")
    last_ts = float(last_ts_raw) if last_ts_raw else 0.0
    newest_ts = last_ts

    # Site-wide posts mentioning T-Mobile
    for sub in reddit.subreddit("all").search("T-Mobile OR TMobile", limit=limit_search):
        created = float(getattr(sub, "created_utc", 0) or 0)
        if created <= last_ts:
            continue
        newest_ts = max(newest_ts, created)

        title = sub.title or ""
        body = sub.selftext or ""
        text = f"{title} {body}".strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        rows.append(Insight(
            source="reddit_post",
            author=str(sub.author) if sub.author else None,
            text=text[:2000],
            timestamp=datetime.datetime.utcfromtimestamp(created).isoformat(),
            url=sub.url,  # unique
            location=extract_location(text),
            score_raw=float(getattr(sub, "score", 0) or 0),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"]
        ))

    # Latest comments in r/TMobile
    for c in reddit.subreddit("TMobile").comments(limit=limit_comments):
        created = float(getattr(c, "created_utc", 0) or 0)
        if created <= last_ts:
            continue
        newest_ts = max(newest_ts, created)

        text = (c.body or "").strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        rows.append(Insight(
            source="reddit_comment",
            author=str(c.author) if c.author else None,
            text=text[:2000],
            timestamp=datetime.datetime.utcfromtimestamp(created).isoformat(),
            url=f"https://reddit.com{c.permalink}",  # unique
            location=extract_location(text),
            score_raw=float(getattr(c, "score", 0) or 0),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"]
        ))

    # Store rows (INSERT OR IGNORE by unique url)
    conn = db()
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT OR IGNORE INTO insights (
                source, author, text, timestamp, url, location,
                score_raw, sentiment_label, sentiment_score, topic, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location,
            r.score_raw, r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()

    if newest_ts > last_ts:
        set_checkpoint("reddit", str(newest_ts))

    return IngestResult(source="reddit_post", inserted=len(rows))


# ============================================================
# 8) PLAY STORE (DELTA MODE)
# ============================================================
def ingest_playstore(count=200) -> IngestResult:
    raw, _ = reviews(
        "com.tmobile.pr.mytmobile",
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=count
    )

    last_ts_raw = get_checkpoint("playstore")
    last_ts = float(last_ts_raw) if last_ts_raw else 0.0
    newest = last_ts
    rows: List[Insight] = []

    for r in raw:
        dt = r.get("at")
        if not dt:
            continue
        ts = float(dt.timestamp())
        if ts <= last_ts:
            continue
        newest = max(newest, ts)

        text = (r.get("content") or "").strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        rows.append(Insight(
            source="playstore",
            author=r.get("userName") or None,
            text=text[:2000],
            timestamp=dt.isoformat(),
            url="https://play.google.com/store/apps/details?id=com.tmobile.pr.mytmobile",  # constant but INSERT OR IGNORE ensures 1 per timestamp combo
            location=extract_location(text),
            score_raw=float(r.get("score") or 0),
            sentiment_label=sentiment,
            sentiment_score=score,
            topic=meta["topic"],
            severity=meta["severity"]
        ))

    conn = db()
    cur = conn.cursor()
    for r in rows:
        # Use (url + timestamp + author + text prefix) uniqueness indirectly via IGNORE on url;
        # If you want stricter dedupe for Play Store, consider adding a hash column.
        cur.execute("""
            INSERT OR IGNORE INTO insights (
                source, author, text, timestamp, url, location,
                score_raw, sentiment_label, sentiment_score, topic, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.source, r.author, r.text, r.timestamp, r.url, r.location,
            r.score_raw, r.sentiment_label, r.sentiment_score, r.topic, r.severity
        ))
    conn.commit()
    conn.close()

    if newest > last_ts:
        set_checkpoint("playstore", str(newest))

    return IngestResult(source="playstore", inserted=len(rows))


# ============================================================
# 9) ROUTES
# ============================================================
@app.get("/health")
def health():
    return {"ok": True, "db": os.path.abspath(DB_PATH)}

@app.post("/ingest/twitter")
async def api_ingest_twitter(max_pages: int = 5):
    try:
        return await ingest_twitter(max_pages=max_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/reddit")
def api_ingest_reddit(limit_search: int = 50, limit_comments: int = 200):
    try:
        return ingest_reddit(limit_search=limit_search, limit_comments=limit_comments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/playstore")
def api_ingest_playstore(count: int = 200):
    try:
        return ingest_playstore(count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/all")
async def api_ingest_all():
    try:
        t_task = asyncio.create_task(ingest_twitter())
        loop = asyncio.get_event_loop()
        r_res = await loop.run_in_executor(None, ingest_reddit, 50, 200)
        p_res = await loop.run_in_executor(None, ingest_playstore, 200)
        t_res = await t_task
        return {"ingested": [r_res.model_dump(), p_res.model_dump(), t_res.model_dump()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    return [Insight(
        source=row[0], author=row[1], text=row[2], timestamp=row[3],
        url=row[4], location=row[5], score_raw=row[6],
        sentiment_label=row[7], sentiment_score=row[8],
        topic=row[9], severity=row[10]
    ) for row in rows]

@app.get("/insights/summary")
def api_summary():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT sentiment_label, COUNT(*) FROM insights GROUP BY sentiment_label")
    sents = dict(cur.fetchall())
    cur.execute("SELECT topic, COUNT(*) FROM insights GROUP BY topic")
    topics = dict(cur.fetchall())
    cur.execute("SELECT severity, COUNT(*) FROM insights GROUP BY severity")
    sevs = dict(cur.fetchall())
    conn.close()
    return {
        "sentiment_counts": sents,
        "topic_counts": topics,
        "severity_counts": sevs
    }

@app.post("/chat")
def chat(question: dict):
    q = (question or {}).get("question", "").strip()
    if not q:
        return {"answer": "Please ask a question."}

    # Pull recent insights for context
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT source, sentiment_label, topic, severity, text, timestamp
        FROM insights
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()

    # Build compact context to keep tokens low
    lines = []
    for src, sent, top, sev, txt, ts in rows:
        snippet = (txt or "").replace("\n", " ").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        lines.append(f"[{src} | {ts} | {sent} | {top} | {sev}] {snippet}")
    context = "\n".join(lines[:180])

    prompt = f"""
You are a T-Mobile Customer Insights Analyst AI.

Use ONLY the insights below to answer the user's question. Be concise, cite patterns, and avoid speculation.

### INSIGHTS ###
{context}

### QUESTION ###
{q}

Provide a clear, structured answer (bullets or short paragraphs). If relevant, include rough counts (e.g., "about 10 mentions") inferred from the context provided.
"""

    try:
        resp = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
            max_output_tokens=300,
        )
        return {"answer": resp.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
