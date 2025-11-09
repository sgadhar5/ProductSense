import sqlite3

DB_PATH = "pulseai.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

posts = [
    ("Love T-Mobile’s new 5G speed improvements!", "positive", 0.95),
    ("Customer service was frustrating yesterday.", "negative", 0.2),
    ("Signal quality is decent, but could be better.", "neutral", 0.6),
    ("Really appreciate the new plans.", "positive", 0.9)
]

cursor.executemany("INSERT INTO posts (text, sentiment, score) VALUES (?, ?, ?)", posts)
conn.commit()
conn.close()
print("✅ Inserted sample posts!")
