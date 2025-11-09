import { useEffect, useState } from "react";
import "../styles/trending.css";

export default function TrendingIssues({ activeTopic, onSelectTopic }) {
  const [topics, setTopics] = useState([]);

  const loadTopics = () => {
    fetch("http://localhost:8000/insights/trending_issues")
      .then(r => r.json())
      .then(setTopics)
      .catch(console.error);
  };

  useEffect(() => {
    loadTopics(); // initial

    const interval = setInterval(loadTopics, 20000); // refresh every 20s

    return () => clearInterval(interval);
  }, []);

  if (!topics.length) return null;

  return (
    <div className="panel">
      <div className="panel-title">Trending Issues (24h)</div>

      <ul className="trend-list">
        {topics.map(t => (
          <li
            key={t.topic}
            className={`trend-item ${activeTopic === t.topic ? "active" : ""}`}
            onClick={() => onSelectTopic(t.topic)}
          >
            <span className="topic">{t.topic}</span>
            <span className="count">{t.count}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
