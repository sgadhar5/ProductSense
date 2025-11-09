# routes/insights.py
from fastapi import APIRouter
from services.analytics import compute_overview, keyword_summary
from services.summarizer import summarize_trends

router = APIRouter()

@router.get("/insights")
def get_insights():
    overview = compute_overview()
    topics = keyword_summary()
    summary = summarize_trends({
        "average_sentiment": overview["avg_sentiment"],
        "post_count": overview["count"],
        "topics": topics,
    })
    return {"overview": overview, "topics": topics, "summary": summary}
