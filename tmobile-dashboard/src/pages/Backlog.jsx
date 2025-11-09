import BacklogCardNavigator from "../components/BacklogCardNavigator";
import "../styles/backlog.css";

export default function Backlog() {
  return (
    <div className="page-container backlog-page">
      <div className="page-inner"> {/* <-- new centering wrapper */}
        <h1 className="page-title">📝 Product Backlog</h1>
        <p className="page-subtitle">
          Review, annotate, and resolve customer-reported issues.
        </p>

        {/* Hard center the card area */}
        <div className="backlog-card-wrapper">
          <BacklogCardNavigator />
        </div>
      </div>
    </div>
  );
}
