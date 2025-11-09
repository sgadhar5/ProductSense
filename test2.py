import pandas as pd, torch
from tqdm import tqdm
from google_play_scraper import reviews, Sort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------
# Load sentiment model
# --------------------------
print("🔧 Loading RoBERTa sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
LABELS = ["Negative", "Neutral", "Positive"]

def analyze_sentiment(text):
    text = text.strip()
    if not text:
        return "Neutral", 0.0
    text = text[:1000]  # prevent overly long reviews
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])

# --------------------------
# Pull Play Store reviews
# --------------------------
print("📱 Fetching reviews for 'T-Mobile' app...")
all_reviews, _ = reviews(
    "com.tmobile.pr.mytmobile",  # official T-Mobile app package name
    lang="en",
    country="us",
    sort=Sort.NEWEST,
    count=200  # pull 200 latest reviews
)

print(f"📦 Collected {len(all_reviews)} reviews")

# --------------------------
# Sentiment analysis
# --------------------------
rows = []
for r in tqdm(all_reviews, desc="Analyzing sentiment"):
    sentiment, score = analyze_sentiment(r["content"])
    rows.append({
        "source": "play_store",
        "userName": r.get("userName", ""),
        "score_rating": r.get("score", None),
        "at": r.get("at"),
        "text": r.get("content", "")[:500],
        "sentiment": sentiment,
        "confidence": score
    })

df = pd.DataFrame(rows)
df.to_csv("playstore_tmobile_sentiment.csv", index=False)
print("\n💾 Saved to playstore_tmobile_sentiment.csv")

# --------------------------
# Summarize sentiment
# --------------------------
summary = df["sentiment"].value_counts(normalize=True) * 100
print("\n📊 Sentiment Summary (Play Store):")
for s in ["Positive", "Neutral", "Negative"]:
    if s in summary:
        print(f"  {s}: {summary[s]:.1f}%")
