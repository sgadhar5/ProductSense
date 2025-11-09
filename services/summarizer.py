# backend/services/summarizer.py
import os
from openai import OpenAI

# Load API key from environment (.env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def summarize_trends(findings: dict) -> str:
    """
    Summarize metrics using OpenAI 2.x API.
    If no key is set, return a mock summary.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "Mock summary: Sentiment remains positive overall; "
            "billing and coverage are frequent topics."
        )

    prompt = f"You are PulseAI. Summarize these findings in 3 sentences:\n{findings}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Be concise and data-driven."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
