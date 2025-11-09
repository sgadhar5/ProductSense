############################################################
#  T-Mobile Customer Happiness Platform — Backend (v4.1)
#  Full single-file backend with:
#   - Twitter, Reddit POSTS + COMMENTS, Play Store ingestion (deduped)
#   - RoBERTa sentiment (CardiffNLP)
#   - GPT topic + severity + location extraction
#   - Full Backlog system (CRUD + Kanban + auto-generate)
#   - Insights API + Chat API
############################################################

import os
import json
import sqlite3
import datetime
import asyncio
from typing import List, Optional, Literal, Dict, Any
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ML
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# Reddit
import praw

# Twitter (Twikit)
from twikit import Client as TwClient

# Play Store
from google_play_scraper import reviews, Sort


############################################################
#  CONFIG
############################################################
load_dotenv()

app = FastAPI(title="Customer Happiness API", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
oaiclient = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = os.getenv("DB_PATH", "pulseai.db")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "tmobile_insights")

TWITTER_COOKIES_PATH = os.getenv("TWITTER_COOKIES_PATH", "cookies_twikit.json")
TWITTER_WORKING_USERNAME = os.getenv("TWITTER_WORKING_USERNAME", "hackutd2025")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


############################################################
#  DB + SCHEMA
############################################################
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
        severity TEXT,
        content_hash TEXT UNIQUE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS checkpoints (
        source TEXT PRIMARY KEY,
        last_value TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS backlog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        insight_id INTEGER,
        summary TEXT NOT NULL,
        description TEXT NOT NULL,
        topic TEXT,
        severity TEXT,
        status TEXT DEFAULT 'todo',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(insight_id)
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


############################################################
#  MODELS
############################################################
SourceType = Literal["twitter","reddit_post","reddit_comment","playstore"]

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
    topic: str
    severity: str



class BacklogItem(BaseModel):
    insight_id: Optional[int] = None
    summary: str
    description: str
    topic: str
    severity: str
    status: str = "todo"
    notes: Optional[str] = ""

class BacklogUpdate(BaseModel):
    summary: Optional[str]
    description: Optional[str]
    topic: Optional[str]
    severity: Optional[str]
    status: Optional[Literal["todo","doing","done"]]
    notes: Optional[str]


class BacklogUpdate(BaseModel):
    summary: Optional[str]
    description: Optional[str]
    topic: Optional[str]
    severity: Optional[str]
    status: Optional[Literal["todo","doing","done"]]


############################################################
#  UTILITIES
############################################################
def compute_hash(text: str, timestamp: str, source: str):
    return hashlib.sha256(f"{source}:{timestamp}:{text}".encode()).hexdigest()


############################################################
#  SENTIMENT MODEL
############################################################
print("Loading RoBERTa sentiment model...")
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


############################################################
#  GPT Topic + Severity
############################################################
TOPIC_ENUM = "billing | 5G | outage | pricing | customer_service | devices | trade-ins | autopay | app | general"

def classify_topic_severity(text: str):
    prompt = f"""
Return ONLY JSON with:
{{
  "topic": one of [{TOPIC_ENUM}],
  "severity": "low" | "medium" | "high"
}}

Rules quick:
- "down/no service/outage/can't connect" → topic: outage or 5G; severity: high
- billing/payment/autopay failures → billing/autopay; severity: high
- angry language → bump severity ≥ medium
- unsure → medium

Text:
\"\"\"{text[:800]}\"\"\"
"""
    try:
        r = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=120
        )
        return json.loads(r.output_text)
    except:
        return {"topic": "general", "severity": "medium"}


############################################################
#  GPT Location Extraction
############################################################
def extract_location(text: str):
    prompt = f"""
Return ONLY JSON as: {{"location": <string or null>}}
Extract a geographic location if explicitly present.

\"\"\"{text[:800]}\"\"\"
"""
    try:
        r = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=60
        )
        loc = json.loads(r.output_text).get("location")
        return loc if (isinstance(loc, str) and loc.strip()) else None
    except:
        return None


############################################################
#  GPT Backlog Summary Generator
############################################################
def draft_backlog_fields(insight: Dict[str, Any]):
    text = insight.get("text", "")
    topic = insight.get("topic", "general")
    severity = insight.get("severity", "medium")
    source = insight.get("source", "")
    url = insight.get("url", "")

    prompt = f"""
You MUST return valid JSON only. No commentary. No markdown. No notes.

Return exactly this format:
{{
  "summary": "string",
  "description": "string",
  "status": "todo"
}}

Strong rules:
- SUMMARY must be a clear human action item.
- NEVER use generic summaries like "issue related to app".
- DESCRIPTION must be bullet points describing problem, impact, source.
- Do NOT include markdown.
- Do NOT wrap in code fences.

User text:
{text[:800]}

Topic: {topic}
Severity: {severity}
Source: {source}
URL: {url}

Respond ONLY with JSON.
"""

    try:
        r = oaiclient.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
            max_output_tokens=200
        )

        output = r.output_text.strip()

        # Remove accidental code fencing
        if output.startswith("```"):
            output = output.strip("```").replace("json", "").strip()

        data = json.loads(output)

        return {
            "summary": data.get("summary", f"{topic.title()} issue")[:120],
            "description": data.get("description", "- user report"),
            "status": data.get("status", "todo")
        }

    except Exception as e:
        print("draft_backlog_fields() fallback due to error:", e)
        return {
            "summary": f"{topic.title()} issue"[:120],
            "description": f"- Severity: {severity}\n- Source: {source}\n- URL: {url}",
            "status": "todo"
        }

@app.post("/backlog/regenerate_clean")
def backlog_regenerate_clean():
    """
    Deletes ALL backlog items and regenerates them using the improved
    GPT-backed summary generator + classification logic.
    """
    conn = db()
    cur = conn.cursor()

    # 1. Delete all backlog items
    cur.execute("DELETE FROM backlog")
    conn.commit()

    print("🧹 Cleared backlog table")

    # 2. Re-run auto-generation
    from fastapi.testclient import TestClient
    client = TestClient(app)

    resp = client.post("/backlog/auto_from_insights")
    conn.close()

    print("⚡️ Regenerated backlog:", resp.json())
    return {
        "status": "ok",
        "message": "Backlog cleared and regenerated.",
        "details": resp.json()
    }

@app.get("/insights/sentiment_score")
def insights_sentiment_score():
    conn = db()
    cur = conn.cursor()

    # Count by sentiment
    cur.execute("""
        SELECT sentiment_label, COUNT(*)
        FROM insights
        GROUP BY sentiment_label
    """)
    sentiment_counts = {row[0]: row[1] for row in cur.fetchall()}

    # Count high-severity issues
    cur.execute("""
        SELECT COUNT(*)
        FROM insights
        WHERE severity='high'
    """)
    high_severity = cur.fetchone()[0]

    # Count outages
    cur.execute("""
        SELECT COUNT(*)
        FROM insights
        WHERE topic='outage'
           OR topic='5G'
    """)
    outages = cur.fetchone()[0]

    conn.close()

    total = sum(sentiment_counts.values()) or 1
    overall_score = (
        (sentiment_counts.get("Positive", 0) - sentiment_counts.get("Negative", 0))
        / total
    )

    return {
        "overall": round(overall_score, 3),
        "high_severity": high_severity,
        "outages": outages
    }

############################################################
#  TWITTER INGEST
############################################################
async def twitter_client() -> TwClient:
    client = TwClient("en-US")
    client.load_cookies(TWITTER_COOKIES_PATH)
    await client.get_user_by_screen_name(TWITTER_WORKING_USERNAME)
    return client


async def ingest_twitter(max_pages=3):
    client = await twitter_client()

    last_id = int(get_checkpoint("twitter") or 0)

    q = '("T-Mobile") OR "TMobile" OR "T Mobile" -arena -concert -tickets'
    tweets = await client.search_tweet(q, "Latest")

    all_tweets = list(tweets)
    page = 1
    while page < max_pages:
        try:
            nxt = await tweets.next()
            if not nxt:
                break
            all_tweets.extend(nxt)
            page += 1
        except:
            break

    new_rows = []
    max_seen = last_id

    for t in all_tweets:
        tid = int(getattr(t, "id", 0) or 0)
        if tid <= last_id:
            continue

        text = (t.text or "").strip()
        if not text:
            continue

        sentiment, s_score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        ts = getattr(t, "created_at", None)
        if isinstance(ts, datetime.datetime):
            ts_iso = ts.isoformat()
        elif isinstance(ts, str):
            ts_iso = ts
        else:
            ts_iso = datetime.datetime.utcnow().isoformat()

        new_rows.append({
            "source": "twitter",
            "author": getattr(getattr(t, "user", None), "name", None),
            "text": text,
            "timestamp": ts_iso,
            "url": f"https://x.com/{getattr(getattr(t, 'user', None), 'screen_name', 'user')}/status/{tid}",
            "location": extract_location(text),
            "score_raw": float(getattr(t, "favorite_count", 0) or 0) + float(getattr(t, "retweet_count", 0) or 0),
            "sentiment_label": sentiment,
            "sentiment_score": s_score,
            "topic": meta["topic"],
            "severity": meta["severity"]
        })

        max_seen = max(max_seen, tid)

    # Insert
    conn = db()
    cur = conn.cursor()
    inserted = 0

    for r in new_rows:
        h = compute_hash(r["text"], r["timestamp"], r["source"])
        cur.execute("""
            INSERT OR IGNORE INTO insights
            (source, author, text, timestamp, url, location, score_raw,
             sentiment_label, sentiment_score, topic, severity, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["source"], r["author"], r["text"], r["timestamp"], r["url"],
            r["location"], r["score_raw"], r["sentiment_label"],
            r["sentiment_score"], r["topic"], r["severity"], h
        ))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()

    if max_seen > last_id:
        set_checkpoint("twitter", str(max_seen))

    return {"source": "twitter", "inserted": inserted}


############################################################
#  REDDIT INGEST — POSTS
############################################################
def reddit_client():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )


def ingest_reddit_posts(limit_search=60):
    reddit = reddit_client()

    last_ts = float(get_checkpoint("reddit_post") or 0.0)
    newest = last_ts

    query = 'T-Mobile OR "T Mobile" OR TMobile OR tmobile'
    rows = []

    for sub in reddit.subreddit("all").search(query, sort="new", limit=limit_search):
        created = float(getattr(sub, "created_utc", 0.0))
        if created <= last_ts:
            continue
        newest = max(newest, created)

        title = sub.title or ""
        body = sub.selftext or ""
        text = f"{title} {body}".strip()
        if not text:
            continue

        sentiment, score = analyze_sentiment(text)
        meta = classify_topic_severity(text)
        ts_iso = datetime.datetime.utcfromtimestamp(created).isoformat()

        rows.append({
            "source": "reddit_post",
            "author": str(sub.author) if sub.author else None,
            "text": text,
            "timestamp": ts_iso,
            "url": f"https://www.reddit.com{sub.permalink}" if getattr(sub, "permalink", None) else sub.url,
            "location": extract_location(text),
            "score_raw": float(getattr(sub, "score", 0) or 0),
            "sentiment_label": sentiment,
            "sentiment_score": score,
            "topic": meta["topic"],
            "severity": meta["severity"]
        })

    # Insert
    conn = db()
    cur = conn.cursor()
    inserted = 0

    for r in rows:
        h = compute_hash(r["text"], r["timestamp"], r["source"])
        cur.execute("""
            INSERT OR IGNORE INTO insights
            (source, author, text, timestamp, url, location, score_raw,
             sentiment_label, sentiment_score, topic, severity, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["source"], r["author"], r["text"], r["timestamp"], r["url"],
            r["location"], r["score_raw"], r["sentiment_label"],
            r["sentiment_score"], r["topic"], r["severity"], h
        ))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()

    if newest > last_ts:
        set_checkpoint("reddit_post", str(newest))

    return {"source": "reddit_post", "inserted": inserted}


############################################################
#  REDDIT INGEST — COMMENTS (from matched posts)
############################################################
def ingest_reddit_comments(limit_posts=40, max_comments_per_post=60):
    reddit = reddit_client()

    last_c_ts = float(get_checkpoint("reddit_comment") or 0.0)
    newest_c = last_c_ts

    query = 'T-Mobile OR "T Mobile" OR TMobile OR tmobile'
    # Get recent matching posts, then scrape their comments
    posts = list(reddit.subreddit("all").search(query, sort="new", limit=limit_posts))

    rows = []

    for post in posts:
        try:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()
        except Exception:
            continue

        count = 0
        for c in comments:
            # Stop per-post cap
            if count >= max_comments_per_post:
                break
            try:
                c_created = float(getattr(c, "created_utc", 0.0))
            except Exception:
                continue
            if c_created <= last_c_ts:
                continue
            newest_c = max(newest_c, c_created)

            text = (getattr(c, "body", "") or "").strip()
            if not text:
                continue

            # MUST mention T-Mobile-ish content to keep relevance (cheap filter)
            # You can relax this if desired
            low = text.lower()
            if not any(k in low for k in ["t-mobile", "tmobile", "t mobile", "magenta", "5g", "outage", "t life", "billing"]):
                continue

            sentiment, s = analyze_sentiment(text)
            meta = classify_topic_severity(text)
            ts_iso = datetime.datetime.utcfromtimestamp(c_created).isoformat()
            url = f"https://www.reddit.com{getattr(c, 'permalink', '')}"

            rows.append({
                "source": "reddit_comment",
                "author": str(getattr(c, "author", None)) if getattr(c, "author", None) else None,
                "text": text,
                "timestamp": ts_iso,
                "url": url,
                "location": extract_location(text),
                "score_raw": float(getattr(c, "score", 0) or 0),
                "sentiment_label": sentiment,
                "sentiment_score": s,
                "topic": meta["topic"],
                "severity": meta["severity"]
            })
            count += 1

    # Insert
    conn = db()
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        h = compute_hash(r["text"], r["timestamp"], r["source"])
        cur.execute("""
            INSERT OR IGNORE INTO insights
            (source, author, text, timestamp, url, location, score_raw,
             sentiment_label, sentiment_score, topic, severity, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["source"], r["author"], r["text"], r["timestamp"], r["url"],
            r["location"], r["score_raw"], r["sentiment_label"],
            r["sentiment_score"], r["topic"], r["severity"], h
        ))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()

    if newest_c > last_c_ts:
        set_checkpoint("reddit_comment", str(newest_c))

    return {"source": "reddit_comment", "inserted": inserted}


############################################################
#  PLAY STORE INGEST
############################################################
def ingest_playstore(count=150):
    raw, _ = reviews(
        "com.tmobile.pr.mytmobile",
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=count
    )

    last_ts = float(get_checkpoint("playstore") or 0.0)
    newest = last_ts

    rows = []

    for r in raw:
        dt = r.get("at")
        if not dt:
            continue

        ts = float(dt.timestamp())
        if ts <= last_ts:
            continue
        newest = max(newest, ts)

        text = r.get("content", "")
        if not text:
            continue

        sentiment, s_score = analyze_sentiment(text)
        meta = classify_topic_severity(text)

        rows.append({
            "source": "playstore",
            "author": r.get("userName"),
            "text": text,
            "timestamp": dt.isoformat(),
            "url": "https://play.google.com/store/apps/details?id=com.tmobile.pr.mytmobile",
            "location": extract_location(text),
            "score_raw": float(r.get("score") or 0),
            "sentiment_label": sentiment,
            "sentiment_score": s_score,
            "topic": meta["topic"],
            "severity": meta["severity"]
        })

    conn = db()
    cur = conn.cursor()
    inserted = 0

    for r in rows:
        h = compute_hash(r["text"], r["timestamp"], r["source"])
        cur.execute("""
            INSERT OR IGNORE INTO insights
            (source, author, text, timestamp, url, location, score_raw,
             sentiment_label, sentiment_score, topic, severity, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["source"], r["author"], r["text"], r["timestamp"], r["url"],
            r["location"], r["score_raw"], r["sentiment_label"],
            r["sentiment_score"], r["topic"], r["severity"], h
        ))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()

    if newest > last_ts:
        set_checkpoint("playstore", str(newest))

    return {"source": "playstore", "inserted": inserted}


############################################################
#  BACKLOG ENDPOINTS (CRUD + AUTO)
############################################################
@app.get("/backlog")
def backlog_list():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, insight_id, summary, description, topic, severity, status,
               created_at, updated_at
        FROM backlog
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "insight_id": r[1],
            "summary": r[2],
            "description": r[3],
            "topic": r[4],
            "severity": r[5],
            "status": r[6],
            "created_at": r[7],
            "updated_at": r[8],
        }
        for r in rows
    ]


@app.get("/backlog/kanban")
def backlog_kanban():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, insight_id, summary, topic, severity, status
        FROM backlog
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()

    board = {"todo": [], "doing": [], "done": []}
    for r in rows:
        item = {"id": r[0], "insight_id": r[1], "summary": r[2],
                "topic": r[3], "severity": r[4]}
        board[r[5]].append(item)
    return board


@app.post("/backlog")
def backlog_create(item: BacklogItem):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO backlog (insight_id, summary, description, topic, severity, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        item.insight_id,
        item.summary,
        item.description,
        item.topic,
        item.severity,
        item.status,
        item.notes,
    ))
    conn.commit()

    return {"created": True}


@app.patch("/backlog/{item_id}")
def backlog_update(item_id: int, data: dict):
    conn = sqlite3.connect("db.sqlite")
    cur = conn.cursor()

    fields = []
    values = []

    if "status" in data:
        fields.append("status = ?")
        values.append(data["status"])

    if "notes" in data:
        fields.append("notes = ?")
        values.append(data["notes"])

    # Require at least one field
    if not fields:
        return {"error": "No valid fields to update"}

    values.append(item_id)

    cur.execute(
        f"UPDATE backlog SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        values
    )

    conn.commit()
    conn.close()

    return {"success": True}


# ======== ANALYTICS ENDPOINTS ========

from datetime import datetime, timedelta

def _to_iso(ts: float | str) -> str:
    if isinstance(ts, str):
        return ts
    return datetime.utcfromtimestamp(ts).isoformat()

@app.get("/insights/sentiment_score")
def insights_sentiment_score():
    """
    Returns:
      {
        overall: float (0..1),
        delta_24h: float (difference vs previous 24h),
        high_severity: int (last 24h),
        outages: int (last 24h),
        trend: [24 floats]  # hourly sentiment over last 24h
      }
    """
    conn = db()
    cur = conn.cursor()

    now = datetime.utcnow()
    t0 = (now - timedelta(hours=24)).isoformat()
    tprev0 = (now - timedelta(hours=48)).isoformat()
    tprev1 = (now - timedelta(hours=24)).isoformat()

    # Overall (last 24h)
    cur.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN sentiment_label='Positive' THEN 1
                        WHEN sentiment_label='Neutral' THEN 0.5
                        ELSE 0 END)
        FROM insights
        WHERE timestamp >= ?
    """, (t0,))
    total_24h, pos_like_24h = cur.fetchone()
    total_24h = total_24h or 0
    pos_like_24h = pos_like_24h or 0.0
    overall = round(pos_like_24h / total_24h, 3) if total_24h else None

    # Previous 24h window
    cur.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN sentiment_label='Positive' THEN 1
                        WHEN sentiment_label='Neutral' THEN 0.5
                        ELSE 0 END)
        FROM insights
        WHERE timestamp >= ? AND timestamp < ?
    """, (tprev0, tprev1))
    total_prev, pos_like_prev = cur.fetchone()
    total_prev = total_prev or 0
    pos_like_prev = pos_like_prev or 0.0
    prev_overall = (pos_like_prev / total_prev) if total_prev else None
    delta = round((overall - prev_overall), 3) if (overall is not None and prev_overall is not None) else None

    # High severity + outage counts (last 24h)
    cur.execute("SELECT COUNT(*) FROM insights WHERE severity='high' AND timestamp >= ?", (t0,))
    high_sev = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM insights WHERE topic='outage' AND timestamp >= ?", (t0,))
    outages = cur.fetchone()[0] or 0

    # Hourly trend (24 buckets)
    trend = []
    for h in range(24):
        start = (now - timedelta(hours=24-h))
        end   = (now - timedelta(hours=23-h))
        cur.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN sentiment_label='Positive' THEN 1
                            WHEN sentiment_label='Neutral' THEN 0.5
                            ELSE 0 END)
            FROM insights
            WHERE timestamp >= ? AND timestamp < ?
        """, (start.isoformat(), end.isoformat()))
        c, pl = cur.fetchone()
        c = c or 0
        pl = pl or 0.0
        trend.append(round(pl / c, 3) if c else None)

    conn.close()
    return {
        "overall": overall,
        "delta_24h": delta,
        "high_severity": high_sev,
        "outages": outages,
        "trend": trend
    }


@app.get("/insights/source_counts")
def insights_source_counts():
    """
    Returns total + last 24h counts by source.
    """
    conn = db()
    cur = conn.cursor()
    now = datetime.utcnow()
    t0 = (now - timedelta(hours=24)).isoformat()

    sources = ["twitter", "reddit_post", "reddit_comment", "playstore"]
    data = {}
    for s in sources:
        cur.execute("SELECT COUNT(*) FROM insights WHERE source=?", (s,))
        total = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM insights WHERE source=? AND timestamp >= ?", (s, t0))
        last24 = cur.fetchone()[0] or 0
        data[s] = {"total": total, "last24h": last24}
    conn.close()
    return data


@app.get("/insights/trending_issues")
def insights_trending_issues(limit: int = 3):
    """
    Top topics last 24h by volume.
    """
    conn = db()
    cur = conn.cursor()
    t0 = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    cur.execute("""
        SELECT topic, COUNT(*) as c
        FROM insights
        WHERE timestamp >= ?
        GROUP BY topic
        ORDER BY c DESC
        LIMIT ?
    """, (t0, limit))
    rows = cur.fetchall()
    conn.close()

    return [{"topic": r[0], "count": r[1]} for r in rows if r[0]]


@app.get("/insights/negative_recent")
def insights_negative_recent(limit: int = 12):
    """
    Recent negative/high-severity/outage items for the spotlight strip.
    """
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT source, sentiment_label, topic, severity, text, timestamp, url
        FROM insights
        WHERE sentiment_label='Negative'
           OR severity='high'
           OR topic='outage'
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "source": r[0], "sentiment": r[1], "topic": r[2], "severity": r[3],
            "text": r[4], "timestamp": r[5], "url": r[6]
        })
    return out


@app.get("/insights/outage_regions")
def insights_outage_regions(limit: int = 10):
    """
    Counts by location for outage-tagged items (last 72h).
    """
    conn = db()
    cur = conn.cursor()
    t0 = (datetime.utcnow() - timedelta(hours=72)).isoformat()
    cur.execute("""
        SELECT COALESCE(location, 'Unknown') as loc, COUNT(*) as c
        FROM insights
        WHERE topic='outage' AND timestamp >= ?
        GROUP BY loc
        ORDER BY c DESC
        LIMIT ?
    """, (t0, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"location": r[0], "count": r[1]} for r in rows]

@app.delete("/backlog/{item_id}")
def backlog_delete(item_id: int):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM backlog WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/backlog/auto_from_insights")
def backlog_auto():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
        SELECT i.id, i.text, i.topic, i.severity, i.sentiment_label,
               i.timestamp, i.url, i.source
        FROM insights i
        LEFT JOIN backlog b ON b.insight_id = i.id
        WHERE b.id IS NULL
          AND (i.severity='high' OR i.topic='outage'
               OR (i.sentiment_label='Negative' AND i.topic IN ('billing','app')))
        ORDER BY i.timestamp DESC
        LIMIT 300
    """)
    rows = cur.fetchall()

    created = []

    for row in rows:
        iid, text, topic, severity, sentiment, ts, url, source = row

        drafted = draft_backlog_fields({
            "text": text,
            "topic": topic,
            "severity": severity,
            "sentiment_label": sentiment,
            "url": url,
            "timestamp": ts,
            "source": source,
        })

        try:
            cur.execute("""
                INSERT INTO backlog (insight_id, summary, description, topic, severity, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (iid, drafted["summary"], drafted["description"],
                  topic, severity, drafted["status"]))
            created.append(iid)
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()

    return {"created": len(created), "insights": created}


############################################################
#  INSIGHTS ENDPOINTS
############################################################
@app.get("/insights/recent")
def api_recent(limit: int = 50):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT source, author, text, timestamp, url, location, score_raw,
               sentiment_label, sentiment_score, topic, severity
        FROM insights
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "source": r[0], "author": r[1], "text": r[2],
            "timestamp": r[3], "url": r[4], "location": r[5],
            "score_raw": r[6], "sentiment_label": r[7],
            "sentiment_score": r[8], "topic": r[9], "severity": r[10]
        }
        for r in rows
    ]


@app.get("/insights/grouped")
def api_grouped(limit: int = 50):
    conn = db()
    cur = conn.cursor()

    data = {}
    for src in ["twitter", "reddit_post", "reddit_comment", "playstore"]:
        cur.execute("""
            SELECT source, author, text, timestamp, url, location, score_raw,
                   sentiment_label, sentiment_score, topic, severity
            FROM insights
            WHERE source=?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (src, limit))
        rows = cur.fetchall()

        data[src] = [
            {
                "source": r[0], "author": r[1], "text": r[2],
                "timestamp": r[3], "url": r[4], "location": r[5],
                "score_raw": r[6], "sentiment_label": r[7],
                "sentiment_score": r[8], "topic": r[9], "severity": r[10]
            }
            for r in rows
        ]

    conn.close()
    return data


############################################################
#  CHAT ENDPOINT
############################################################
@app.post("/chat")
def chat(question: dict):
    q = (question or {}).get("question", "").strip()
    if not q:
        return {"answer": "Please ask a question."}

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT source, sentiment_label, topic, severity, text, timestamp
        FROM insights
        ORDER BY timestamp DESC
        LIMIT 300
    """)
    rows = cur.fetchall()
    conn.close()

    context = ""
    for src, s, t, sev, txt, ts in rows:
        snippet = (txt or "").replace("\n"," ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        context += f"[{src} | {ts} | {s} | {t} | {sev}] {snippet}\n"

    prompt = f"""
Use ONLY the insights below to answer the question.

INSIGHTS:
{context}

QUESTION:
{q}

Return a factual, structured answer.
"""

    resp = oaiclient.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.2,
        max_output_tokens=320,
    )
    return {"answer": resp.output_text}


############################################################
#  INGEST ENDPOINTS
############################################################
@app.post("/ingest/twitter")
async def api_ingest_twitter(max_pages: int = 3):
    return await ingest_twitter(max_pages=max_pages)

@app.post("/ingest/reddit_posts")
def api_ingest_reddit_posts(limit_search: int = 60):
    return ingest_reddit_posts(limit_search=limit_search)

@app.post("/ingest/reddit_comments")
def api_ingest_reddit_comments(limit_posts: int = 40, max_comments_per_post: int = 60):
    return ingest_reddit_comments(limit_posts=limit_posts, max_comments_per_post=max_comments_per_post)

@app.post("/ingest/playstore")
def api_ingest_playstore(count: int = 150):
    return ingest_playstore(count=count)

@app.post("/ingest/all")
async def api_ingest_all():
    t_task = asyncio.create_task(ingest_twitter())
    rp = ingest_reddit_posts()
    rc = ingest_reddit_comments()
    ps = ingest_playstore()
    tw = await t_task

    # Auto-create backlog items for severe/outage/billing/app-negatives
    auto = backlog_auto()

    return {
        "twitter": tw,
        "reddit_posts": rp,
        "reddit_comments": rc,
        "playstore": ps,
        "backlog_auto_created": auto
    }

@app.get("/analytics/summary")
def analytics_summary():
    conn = db()
    cur = conn.cursor()

    # Get 300 latest items for sentiment scoring
    cur.execute("""
        SELECT sentiment_label, severity
        FROM insights
        ORDER BY timestamp DESC
        LIMIT 300
    """)
    rows = cur.fetchall()

    conn.close()

    score = 0
    max_score = 0

    for sent, sev in rows:
        # Sentiment weight
        if sent == "Positive":
            s = 1
        elif sent == "Negative":
            s = -1
        else:
            s = 0

        # Severity weight
        if sev == "high":
            s -= 2
        elif sev == "medium":
            s -= 1

        score += s
        max_score += 3  # theoretical max per record

    if max_score == 0:
        final_score = 0
    else:
        final_score = int((score / max_score) * 100)

    return {
        "sentiment_score": final_score
    }

@app.get("/backlog/summary")
def backlog_summary():
    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM backlog WHERE status='todo'")
    todo = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM backlog WHERE status='doing'")
    doing = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM backlog WHERE status='done'")
    done = cur.fetchone()[0]

    cur.execute("""
        SELECT id, summary, topic, severity, status, created_at
        FROM backlog
        ORDER BY created_at DESC
        LIMIT 5
    """)
    rows = cur.fetchall()

    conn.close()

    return {
        "counts": {"todo": todo, "doing": doing, "done": done},
        "latest": [
            {
                "id": r[0],
                "summary": r[1],
                "topic": r[2],
                "severity": r[3],
                "status": r[4],
                "created_at": r[5]
            }
            for r in rows
        ]
    }

############################################################
#  HEALTH
############################################################
@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}
