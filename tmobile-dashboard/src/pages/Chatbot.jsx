import { useState } from "react";
import "../styles/chat.css";
import "../styles/layout.css";

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askChatbot = () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");

    fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    })
      .then((r) => r.json())
      .then((d) => setAnswer(d.answer))
      .finally(() => setLoading(false));
  };

  return (
    <div className="page-container">
      <div className="chat-container">

        <h1 className="chat-title">💬 Insights Chatbot</h1>

        <textarea
          className="chat-input"
          rows="3"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask something like: 'What are the main customer issues today?'"
        />

        <button className="chat-button" onClick={askChatbot}>
          {loading ? "Thinking..." : "Ask"}
        </button>

        <div className="chat-answer-box">
          <h3>Response:</h3>
          <p>{answer || (loading ? "Analyzing..." : "No response yet.")}</p>
        </div>

      </div>
    </div>
  );
}
