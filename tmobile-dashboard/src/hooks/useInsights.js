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
      console.error(e);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return { recent, summary, loading, refresh };
};
