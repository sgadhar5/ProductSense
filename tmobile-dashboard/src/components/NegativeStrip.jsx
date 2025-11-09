import { useEffect, useState } from "react";
import "../styles/negative.css";

export default function NegativeStrip() {
  const [items, setItems] = useState([]);

  const loadItems = () => {
    fetch("http://localhost:8000/insights/negative_recent?limit=12")
      .then(r => r.json())
      .then(setItems)
      .catch(console.error);
  };

  useEffect(() => {
    loadItems(); // initial load

    const interval = setInterval(loadItems, 15000); // refresh every 15 sec

    return () => clearInterval(interval);
  }, []);

  if (!items.length) return null;

  return (
    <div className="neg-strip">
      {items.map((it, i) => (
        <a
          key={i}
          className="neg-card"
          href={it.url || "#"}
          target="_blank"
          rel="noreferrer"
        >
          <div className="neg-top">
            <span className="src">{it.source}</span>
            <span className={`sev ${it.severity}`}>{it.severity}</span>
          </div>

          <div className="neg-body">
            {(it.text || "").slice(0, 120)}
            {(it.text || "").length > 120 ? "…" : ""}
          </div>

          <div className="neg-meta">
            <span className={`sent ${it.sentiment.toLowerCase()}`}>
              {it.sentiment}
            </span>
            <span className="topic">{it.topic}</span>
          </div>
        </a>
      ))}
    </div>
  );
}
