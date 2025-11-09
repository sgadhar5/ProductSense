import "../styles/dashboard.css";

export default function RecentTable({ grouped = {} }) {
  // Safely extract lists from grouped object
  const twitter = grouped.twitter ?? [];
  const redditPosts = grouped.reddit_post ?? [];
  const redditComments = grouped.reddit_comment ?? [];
  const playstore = grouped.playstore ?? [];

  const renderSection = (label, items) => (
    <div className="card" style={{ marginBottom: "2rem" }}>
      <h2>{label}</h2>

      {items.length === 0 ? (
        <p>No data.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Text</th>
              <th>Sentiment</th>
              <th>Topic</th>
              <th>Severity</th>
            </tr>
          </thead>

          <tbody>
            {items.map((r, i) => (
              <tr key={i}>
                <td>{r.text.slice(0, 120)}...</td>
                <td>{r.sentiment_label}</td>
                <td>{r.topic}</td>
                <td>{r.severity}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  return (
    <div>
      {renderSection("Twitter Mentions", twitter)}
      {renderSection("Reddit Posts", redditPosts)}
      {renderSection("Reddit Comments", redditComments)}
      {renderSection("Play Store Reviews", playstore)}
    </div>
  );
}
