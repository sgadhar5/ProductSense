import { useState, useEffect, useRef } from "react";
import "../styles/backlog.css";

export default function BacklogCarousel() {
  const [items, setItems] = useState([]);
  const [index, setIndex] = useState(0);

  const touchStartX = useRef(null);
  const MIN_SWIPE = 60; // px swipe required

  useEffect(() => {
    fetch("http://localhost:8000/backlog")
      .then(r => r.json())
      .then(setItems)
      .catch(console.error);
  }, []);

  const current = items[index] || null;

  // Move left/right
  const prev = () => setIndex(i => (i === 0 ? items.length - 1 : i - 1));
  const next = () => setIndex(i => (i === items.length - 1 ? 0 : i + 1));

  // Touch handling
  const handleTouchStart = (e) => {
    touchStartX.current = e.touches[0].clientX;
  };

  const handleTouchEnd = (e) => {
    if (!touchStartX.current) return;
    const dx = e.changedTouches[0].clientX - touchStartX.current;

    if (dx > MIN_SWIPE) prev();
    if (dx < -MIN_SWIPE) next();

    touchStartX.current = null;
  };

  // Save notes
  const updateNotes = async (notes) => {
    const id = current.id;
    const res = await fetch(`http://localhost:8000/backlog/${id}/update_notes`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes })
    });
    console.log(await res.json());

    setItems((arr) =>
      arr.map((it) =>
        it.id === id ? { ...it, notes } : it
      )
    );
  };

  // Resolve item
  const resolveItem = async () => {
    const id = current.id;
    const res = await fetch(`http://localhost:8000/backlog/${id}/resolve`, {
      method: "PATCH"
    });
    console.log(await res.json());

    setItems((arr) =>
      arr.map((it) =>
        it.id === id ? { ...it, status: "done" } : it
      )
    );
  };

  if (!current) return <h2 className="empty-backlog">No backlog items</h2>;

  return (
    <div className="carousel-wrapper">
      
      {/* Left Arrow */}
      <button className="nav-arrow left" onClick={prev}>❮</button>

      {/* Card */}
      <div
        className="carousel-card"
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        <h2>{current.summary}</h2>

        <div className="badge-row">
          <span className="topic-badge">{current.topic}</span>
          <span className="severity-badge">{current.severity}</span>
          <span className="status-badge">{current.status}</span>
        </div>

        <details className="desc">
          <summary>Description</summary>
          <p>{current.description}</p>
        </details>

        <div className="notes-section">
          <h3>Notes</h3>
          <textarea
            value={current.notes || ""}
            onChange={(e) => updateNotes(e.target.value)}
            placeholder="Add internal notes here..."
          />
        </div>

        <button className="resolve-btn" onClick={resolveItem}>
          Mark as Done
        </button>

        <div className="timestamps">
          <p><b>Created:</b> {current.created_at}</p>
          <p><b>Updated:</b> {current.updated_at}</p>
        </div>
      </div>

      {/* Right Arrow */}
      <button className="nav-arrow right" onClick={next}>❯</button>

    </div>
  );
}
