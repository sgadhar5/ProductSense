import { useEffect, useState } from "react";
import api from "../api/client";

export const useInsights = () => {
  const [recent, setRecent] = useState([]);
  const [summary, setSummary] = useState({});
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const r = await api.get("/insights/recent?limit=50");
      const s = await api.get("/insights/summary");

      setRecent(r.data);
      setSummary(s.data);
      setLoading(false);
    } catch (e) {
      console.error("❌ Failed to refresh insights:", e);
    }
  };

  // Initial load
  useEffect(() => {
    refresh();
  }, []);

  // 🔄 Auto-refresh + auto-ingest every 5 minutes
  useEffect(() => {
    const interval = setInterval(async () => {
      console.log("🔄 Auto-ingesting new data...");

      try {
        // hit backend to ingest new Reddit / Twitter / Play Store items
        await api.post("/ingest/all");
        console.log("✨ New data ingested. Refreshing UI...");

        // reload dashboard data
        await refresh();

      } catch (err) {
        console.error("❌ Auto-ingest failed:", err);
      }
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(interval);
  }, [refresh]);

  return { recent, summary, loading, refresh };
};
