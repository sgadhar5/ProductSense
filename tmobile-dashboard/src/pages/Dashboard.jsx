import { useEffect, useState } from "react";
import RecentTable from "../components/RecentTable";
import "../styles/dashboard.css";
import "../styles/layout.css";
import "../styles/page.css";   // <-- universal centered layout

export default function Dashboard() {
  const [grouped, setGrouped] = useState({});
  const [score, setScore] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/insights/grouped")
      .then((r) => r.json())
      .then(setGrouped)
      .catch(console.error);

    fetch("http://localhost:8000/insights/sentiment_score")
      .then((r) => r.json())
      .then(setScore)
      .catch(console.error);
  }, []);

  return (
    <div className="page">
      <div className="page-inner">

        {/* ======================= */}
        {/* SENTIMENT METRICS ROW */}
        {/* ======================= */}
        <div className="metrics-row">

          <div className="metric-card">
            <div className="metric-title">Overall Sentiment</div>
            <div className="metric-value purple">
              {score?.overall}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-title">High Severity Reports</div>
            <div className="metric-value red">
              {score?.high_severity}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-title">Outage Mentions</div>
            <div className="metric-value orange">
              {score?.outages}
            </div>
          </div>

        </div>

        {/* ======================= */}
        {/* RECENT MENTIONS TABLE */}
        {/* ======================= */}
        <div className="recent-container">
          <h2 className="recent-title">Recent Mentions</h2>
          <RecentTable grouped={grouped} />
        </div>

      </div>
    </div>
  );
}
