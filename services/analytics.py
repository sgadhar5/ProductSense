# services/analytics.py
import os
import sqlite3
from collections import Counter

# 1️⃣ Define DB_PATH first
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "pulseai.db"))

# 2️⃣ Create the table if it doesn't exist
def ensure_tables():
    """Create the posts table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            score REAL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Ensured 'posts' table exists at", DB_PATH)

# Call this once at import
ensure_tables()

# 3️⃣ Main analytics functions
def compute_overview():
    """Compute sentiment averages and counts from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment, score FROM posts")
    data = cursor.fetchall()
    conn.close()

    if not data:
        return {"avg_sentiment": 0.0, "count": 0}

    avg_sentiment = sum(row[1] for row in data) / len(data)
    return {"avg_sentiment": avg_sentiment, "count": len(data)}


def keyword_summary(limit: int = 5):
    """Return top keywords from posts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM posts")
    texts = [row[0] for row in cursor.fetchall()]
    conn.close()

    words = [w.lower() for text in texts for w in text.split() if len(w) > 3]
    common = Counter(words).most_common(limit)
    return [{"topic": word, "count": count} for word, count in common]


