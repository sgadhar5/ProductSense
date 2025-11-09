from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

# Pretrained model fine-tuned on tweets (great for short feedback text)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:
    def __init__(self):
        print("⏳ Loading sentiment model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.labels = ["negative", "neutral", "positive"]
        print("✅ Sentiment model ready.")

    def score(self, text: str):
        """Return sentiment label and numeric score (−1..+1)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs))
        label = self.labels[label_idx]
        # convert to numeric score (−1, 0, +1 weighted by confidence)
        score = (label_idx - 1) * float(probs[label_idx])
        return {
            "label": label,
            "score": round(score, 3),
            "probs": {self.labels[i]: float(probs[i]) for i in range(3)},
        }

# Initialize once for reuse
sentiment_model = SentimentModel()
