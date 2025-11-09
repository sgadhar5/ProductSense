import { Link } from "react-router-dom";
import "../styles/header.css";

export default function Header() {
  return (
    <header className="global-header">
      <div className="header-inner">

        <div className="brand">ProductSense</div>

        <nav className="nav-links">
          <Link to="/">Dashboard</Link>
          <Link to="/backlog">Backlog</Link>
          <Link to="/chat">Chatbot</Link>
        </nav>

      </div>
    </header>
  );
}
