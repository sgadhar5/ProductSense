# routes/analyze.py
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import requests

from models.sentiment_model import sentiment_model
from database.db import get_conn

# Import your scrapers
from scrapers.collector import scrape_all

router = APIRouter()

class AnalyzeInput(BaseModel):
    text: str
    source: str = "manual"
    timestamp: Optional[str] = None

# ✅ Single post analysis (manual or from scraper)
@router.post("/analyze")
def analyze(inp: AnalyzeInput):
    result = sentiment_model.score(inp.text)
    ts = inp.timestamp or datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO feedback (source, text, timestamp, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            (inp.source, inp.text, ts, result["score"], result["label"]),
        )

    return {
        "sentiment": result["label"],
        "score": result["score"],
        "probs": result["probs"],
        "source": inp.source,
        "timestamp": ts,
    }

# ✅ Bulk scraping route — collects live data and pipes to /analyze
@router.post("/scrape")
def scrape_and_analyze(background_tasks: BackgroundTasks, limit: int = 40):
    """
    Scrape T-Mobile related posts from multiple sources (Twitter, Instagram, Play Store),
    analyze sentiment, and store results in the DB.
    """
    background_tasks.add_task(run_scrape_pipeline, limit)
    return {"message": "Scraping started in background"}

def run_scrape_pipeline(limit: int = 40):
    data = scrape_all()[:limit]
    for item in data:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/analyze",
                json={"text": item["text"], "source": item["source"], "timestamp": item["timestamp"]},
                timeout=10
            )
            print(f"✅ {item['source']} post stored ({res.status_code})")
        except Exception as e:
            print(f"⚠️ Failed to analyze {item['source']} post:", e)
