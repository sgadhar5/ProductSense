**ProductSense** – Real-Time Customer Insight Dashboard

**Overview**

PulseAI continuously ingests live customer feedback from multiple online platforms including Twitter, Reddit posts, Reddit comments, and Play Store reviews. 
The system performs sentiment analysis, topic classification, severity scoring, and detects outages, billing problems, and app issues. 
All processed insights feed directly into a Dashboard UI showing executive metrics, trending issues, negative spotlight data, and an automatically generated product backlog with notes, 
resolutions, and filtering.

**Core Features**

--Real-time dashboard that updates automatically.

--Executive sentiment summary including sparklines and 24-hour deltas.

--Source volume cards showing last 24 hours and total counts.

--Trending issues based on activity over the last 24 hours.

--Negative and high-severity spotlight strip.

--Recent mentions grouped by Twitter, Reddit posts, Reddit comments, and Play Store reviews.

--AI-generated product backlog with summaries, severities, topics, and status.

--Backlog card navigation with swipe or arrow controls and editable notes modal.

--Insight chatbot that answers questions about current user sentiment and issues.


**Technology Stack** 

Backend: Python (FastAPI), Hypercorn ASGI server, SQLite, OpenAI GPT models, PRAW for Reddit, Google Play Scraper, Twikit for Twitter.

Frontend: React with Vite, Recharts for sparklines and charts, custom CSS, auto-refreshing data through React hooks.

**Installation**

--Clone the repository.

--Create a Python virtual environment.

--Install backend dependencies via “pip install -r requirements.txt”.

--Create a .env file containing OpenAI keys, Reddit credentials, Twitter cookies path, and database path.

--Start backend with “hypercorn main:app --reload”.

--Start frontend with “npm install” and “npm run dev”.


**Database Schema**
Insights table stores processed feedback including text, source, sentiment label, topic, severity, timestamp, URL, and a hash for deduplication.
Backlog table stores automatically generated tasks including summary, description, topic, severity, notes, status, and timestamps.

**API Endpoints**

Data ingestion: /ingest/all, /ingest/twitter, /ingest/reddit, /ingest/playstore

Insights: /insights/grouped, /insights/sentiment_score, /insights/trending_issues, /insights/negative_recent, /insights/source_counts

Backlog: /backlog, /backlog/{id}, /backlog/auto_from_insights

Chatbot: /chat


**Auto-Refresh Intervals**

--Executive Summary refreshes every 20 seconds.

--Source Volumes refresh every 30 seconds.

--Trending Issues refresh every 30 seconds.

--Negative Strip refreshes every 20 seconds.

--Recent Tabs refresh every 25 seconds.

**Roadmap**

Future additions include region-based outage heatmaps, AI clustering of issues, realtime websocket alerts, historical metrics dashboard, and integration with Jira.



