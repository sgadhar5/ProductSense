import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import "../styles/dashboard.css";

export default function SentimentChart({ data }) {
  const chartData = Object.entries(data || {}).map(([name, value]) => ({
    name,
    value
  }));

  const COLORS = ["#ff4d4d", "#ffcc00", "#33cc33"]; // red, yellow, green

  return (
    <div className="card">
      <h2>Sentiment Breakdown</h2>

      <PieChart width={350} height={260}>
        <Pie
          data={chartData}
          dataKey="value"
          cx="50%"
          cy="50%"
          outerRadius={80}
          label
        >
          {chartData.map((entry, i) => (
            <Cell key={i} fill={COLORS[i]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </div>
  );
}
