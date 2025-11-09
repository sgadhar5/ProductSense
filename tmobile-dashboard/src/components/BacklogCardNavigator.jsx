import { useEffect, useState } from "react";
import Modal from "../pages/Modal.jsx";
import "../styles/backlog.css";

export default function BacklogCardNavigator() {
  const [items, setItems] = useState([]);
  const [index, setIndex] = useState(0);
  const [notesOpen, setNotesOpen] = useState(false);
  const [currentNotes, setCurrentNotes] = useState("");

  useEffect(() => {
    fetch("http://localhost:8000/backlog")
      .then(r => r.json())
      .then(data => setItems(data));
  }, []);

  if (items.length === 0) return <p>No backlog items found.</p>;

  const item = items[index];

  const goPrev = () => setIndex(i => (i > 0 ? i - 1 : i));
  const goNext = () => setIndex(i => (i < items.length - 1 ? i + 1 : i));

  const openNotes = () => {
    setCurrentNotes(item.notes || "");
    setNotesOpen(true);
  };

  const saveNotes = () => {
    fetch(`http://localhost:8000/backlog/${item.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: currentNotes }),
    });

    const updated = [...items];
    updated[index].notes = currentNotes;
    setItems(updated);
    setNotesOpen(false);
  };

  const resolveItem = () => {
    fetch(`http://localhost:8000/backlog/${item.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "done" }),
    });

    // Update UI immediately
    const updated = [...items];
    updated[index].status = "done";
    setItems(updated);

    alert("This issue has been marked as resolved.");
  };

  return (
    <div className="backlog-card-wrapper">

      {/* CARD */}
      <div
        className={
          "backlog-card clean-card " +
          (item.status === "done" ? "card-resolved" : "")
        }
      >
        <h2>{item.summary}</h2>

        <div className="badge-row">
          <span className="badge topic">{item.topic}</span>
          <span className="badge severity">{item.severity}</span>
          <span
            className="badge status"
            style={{ background: item.status === "done" ? "#1bbf72" : "#888" }}
          >
            {item.status.toUpperCase()}
          </span>
        </div>

        <details className="description-box">
          <summary>Description</summary>
          <p>{item.description}</p>
        </details>

        <button className="notes-btn" onClick={openNotes}>
          View / Edit Notes
        </button>

        <button
          className="resolve-btn"
          onClick={resolveItem}
          disabled={item.status === "done"}
        >
          {item.status === "done" ? "Resolved" : "Mark as Resolved"}
        </button>

        <div className="pagination-arrows">
          <button className="arrow-btn" onClick={goPrev} disabled={index === 0}>
            ◀
          </button>
          <span className="index-text">
            {index + 1} / {items.length}
          </span>
          <button
            className="arrow-btn"
            onClick={goNext}
            disabled={index === items.length - 1}
          >
            ▶
          </button>
        </div>
      </div>

      {/* NOTES MODAL */}
      <Modal open={notesOpen} onClose={() => setNotesOpen(false)}>
        <h2>Edit Notes</h2>
        <textarea
          className="modal-textarea"
          value={currentNotes}
          onChange={(e) => setCurrentNotes(e.target.value)}
        />
        <button className="modal-save-btn" onClick={saveNotes}>
          Save Notes
        </button>
      </Modal>

    </div>
  );
}
