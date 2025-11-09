import "../styles/dashboard.css";

export default function RecentTable({ rows }) {
  return (
    <div className="card">
      <h2>Recent Mentions</h2>

      <table className="table">
        <thead>
          <tr>
            <th>Source</th>
            <th>Text</th>
            <th>Sentiment</th>
            <th>Topic</th>
            <th>Severity</th>
          </tr>
        </thead>

        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.source}</td>
              <td>{r.text.slice(0, 120)}...</td>
              <td>{r.sentiment_label}</td>
              <td>{r.topic}</td>
              <td>{r.severity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
