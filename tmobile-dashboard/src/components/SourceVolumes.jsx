import { useEffect, useState } from "react";
import "../styles/sources.css";

const ICON = {
  twitter: "🐦",
  reddit_post: "🔺",
  reddit_comment: "💬",
  playstore: "⭐",
};

export default function SourceVolumes() {
  const [counts, setCounts] = useState(null);

  const loadCounts = () => {
    fetch("http://localhost:8000/insights/source_counts")
      .then((r) => r.json())
      .then(setCounts)
      .catch(console.error);
  };

  useEffect(() => {
    loadCounts(); // initial fetch

    const interval = setInterval(loadCounts, 20000); // refresh every 20s

    return () => clearInterval(interval);
  }, []);

  const cards = counts ? Object.entries(counts) : [];

  return (
    <div className="sources-row">
      {cards.map(([src, val]) => (
        <div key={src} className="source-card">
          <div className="source-top">
            <span className="icon">{ICON[src] || "📡"}</span>
            <span className="src">{src.replace("_", " ")}</span>
          </div>

          <div className="source-metrics">
            <div><b>{val.last24h}</b> last 24h</div>
            <div className="muted">{val.total} total</div>
          </div>
        </div>
      ))}
    </div>
  );
}
