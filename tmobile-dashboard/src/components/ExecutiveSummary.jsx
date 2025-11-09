import { useEffect, useState } from "react";
import Sparkline from "./Sparkline";
import "../styles/exec.css";

export default function ExecutiveSummary() {
  const [score, setScore] = useState(null);

  const loadScore = () => {
    fetch("http://localhost:8000/insights/sentiment_score")
      .then(r => r.json())
      .then(setScore)
      .catch(console.error);
  };

  useEffect(() => {
    // initial load
    loadScore();

    // refresh every 20 seconds
    const interval = setInterval(loadScore, 20000);

    // cleanup
    return () => clearInterval(interval);
  }, []);

  const delta = score?.delta_24h;
  const deltaSign =
    typeof delta === "number" && delta !== 0
      ? delta > 0
        ? "▲"
        : "▼"
      : "";

  const overallColor =
    score?.overall == null
      ? ""
      : score.overall >= 0.6
      ? "good"
      : score.overall >= 0.4
      ? "warn"
      : "bad";

  return (
    <div className="exec-row">
      {/* Overall Sentiment */}
      <div className="exec-card">
        <div className="exec-title">Overall Sentiment</div>
        <div className={`exec-metric ${overallColor}`}>
          {score?.overall != null ? score.overall.toFixed(2) : "—"}

          {deltaSign && (
            <span className={`delta ${delta > 0 ? "up" : "down"}`}>
              {deltaSign} {Math.abs(delta).toFixed(2)}
            </span>
          )}
        </div>

        <div className="sparkline">
          <Sparkline data={score?.trend || []} />
        </div>
      </div>

      {/* High Severity */}
      <div className="exec-card">
        <div className="exec-title">High Severity (24h)</div>
        <div className="exec-metric red">
          {score?.high_severity ?? "—"}
        </div>
      </div>

      {/* Outage Mentions */}
      <div className="exec-card">
        <div className="exec-title">Outage Mentions (24h)</div>
        <div className="exec-metric orange">
          {score?.outages ?? "—"}
        </div>
      </div>
    </div>
  );
}
