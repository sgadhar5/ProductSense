CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- e.g., twitter, instagram, playstore, manual
    text TEXT NOT NULL,            -- raw feedback text
    timestamp TEXT NOT NULL,       -- ISO date string
    sentiment_score REAL,          -- −1.0 .. +1.0 numeric score
    sentiment_label TEXT           -- positive / neutral / negative
);

CREATE INDEX IF NOT EXISTS idx_feedback_time ON feedback(timestamp);
