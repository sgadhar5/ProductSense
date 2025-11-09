import "../styles/dashboard.css";
import { useInsights } from "../hooks/useInsights";
import SentimentChart from "../components/SentimentChart";
import TopicBarChart from "../components/TopicBarChart";
import RecentTable from "../components/RecentTable";

export default function Dashboard() {
  const { recent, summary, loading, refresh } = useInsights();

  if (loading) {
    return <div className="dashboard-container">Loading...</div>;
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>📊 Customer Happiness Dashboard</h1>
        <button className="refresh-btn" onClick={refresh}>Refresh</button>
      </div>

      <div className="grid-2">
        <SentimentChart data={summary.sentiment_counts} />
        <TopicBarChart data={summary.topic_counts} />
      </div>

      <RecentTable rows={recent} />
    </div>
  );
}
