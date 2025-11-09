import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import "../styles/dashboard.css";

export default function TopicBarChart({ data }) {
  const chartData = Object.entries(data || {}).map(([topic, count]) => ({
    topic,
    count
  }));

  return (
    <div className="card">
      <h2>Topic Distribution</h2>

      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData}>
          <XAxis dataKey="topic" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#0066ff" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
