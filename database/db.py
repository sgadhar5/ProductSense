import os, sqlite3
from pathlib import Path

# Use .env value if provided, else default to local file
DB_PATH = Path(os.getenv("DB_PATH", "pulseai.db"))

def get_conn():
    """Return a connection with Row objects for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Run the schema.sql script to ensure tables exist."""
    schema_path = Path(__file__).with_name("schema.sql")
    with get_conn() as conn, open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    print(f"✅ Database initialized at {DB_PATH.resolve()}")
