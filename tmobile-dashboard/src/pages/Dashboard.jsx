import { useState } from "react";
import "../styles/dashboard.css";
import { useInsights } from "../hooks/useInsights";
import SentimentChart from "../components/SentimentChart";
import TopicBarChart from "../components/TopicBarChart";
import RecentTable from "../components/RecentTable";
import api from "../api/client";

export default function Dashboard() {
  const { recent, summary, loading, refresh } = useInsights();
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    try {
      setRefreshing(true);

      // 🔥 Trigger backend ingestion (delta)
      await api.post("/ingest/all");

      // 🔄 Refresh dashboard data
      await refresh();
    } catch (err) {
      console.error(err);
    } finally {
      setRefreshing(false);
    }
  };

  if (loading) {
    return <div className="dashboard-container">Loading...</div>;
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>📊 Customer Happiness Dashboard</h1>

        <button
          className="refresh-btn"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="grid-2">
        <SentimentChart data={summary.sentiment_counts} />
        <TopicBarChart data={summary.topic_counts} />
      </div>

      <RecentTable rows={recent} />
    </div>
  );
}
