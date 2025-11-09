import { useState, useEffect } from "react";
import "../styles/tabs.css";

export default function RecentTabs() {
  const [grouped, setGrouped] = useState({
    twitter: [],
    reddit_post: [],
    reddit_comment: [],
    playstore: []
  });

  const [active, setActive] = useState("twitter");

  // --- refresh function ---
  const loadGrouped = () => {
    fetch("http://localhost:8000/insights/grouped")
      .then((r) => r.json())
      .then(setGrouped)
      .catch(console.error);
  };

  useEffect(() => {
    loadGrouped();  // initial load

    // refresh every 25 seconds
    const interval = setInterval(loadGrouped, 25000);

    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: "twitter", label: "🐦 Twitter" },
    { id: "reddit_post", label: "🔺 Reddit Posts" },
    { id: "reddit_comment", label: "💬 Reddit Comments" },
    { id: "playstore", label: "⭐ Play Store" }
  ];

  return (
    <div className="tabs-container">

      {/* Tabs */}
      <div className="tabs-button-row">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tabs-button ${active === t.id ? "active" : ""}`}
            onClick={() => setActive(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Fixed-height content area */}
      <div className="sections-fixed">

        {/* Twitter */}
        <div className={`section-block ${active === "twitter" ? "active" : ""}`}>
          <h2>Twitter Mentions</h2>
          <Table data={grouped.twitter} />
        </div>

        {/* Reddit posts */}
        <div className={`section-block ${active === "reddit_post" ? "active" : ""}`}>
          <h2>Reddit Posts</h2>
          <Table data={grouped.reddit_post} />
        </div>

        {/* Reddit comments */}
        <div className={`section-block ${active === "reddit_comment" ? "active" : ""}`}>
          <h2>Reddit Comments</h2>
          <Table data={grouped.reddit_comment} />
        </div>

        {/* Play Store */}
        <div className={`section-block ${active === "playstore" ? "active" : ""}`}>
          <h2>Play Store Reviews</h2>
          <Table data={grouped.playstore} />
        </div>

      </div>
    </div>
  );
}


function Table({ data }) {
  if (!data || !data.length) return <p>No data.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Text</th>
          <th>Sentiment</th>
          <th>Topic</th>
          <th>Severity</th>
        </tr>
      </thead>

      <tbody>
        {data.map((row, idx) => (
          <tr key={idx}>
            <td className="table-text-cell">{row.text}</td>
            <td className="table-small-cell">{row.sentiment_label}</td>
            <td className="table-small-cell">{row.topic}</td>
            <td className="table-small-cell">{row.severity}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
