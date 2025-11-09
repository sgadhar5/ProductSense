import { useState } from "react";
import api from "../api/client";   // <-- your axios instance
import "/Users/succhaygadhar/Downloads/ai/tmobile-dashboard/src/styles/chat.css";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! Ask me anything about T-Mobile customer insights." }
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      // Send question to backend
      const response = await api.post("/chat", { question: input });

      // Add bot reply
      const botMsg = { sender: "bot", text: response.data.answer };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error contacting server." }
      ]);
    }

    setInput("");
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.messagesBox}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              ...(m.sender === "user" ? styles.userMsg : styles.botMsg)
            }}
          >
            {m.text}
          </div>
        ))}

        {loading && (
          <div style={{ ...styles.message, ...styles.botMsg }}>Thinking…</div>
        )}
      </div>

      <div style={styles.inputRow}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about insights…"
        />
        <button style={styles.button} onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

//
// 💅 Basic Inline Styles (optional)
//
const styles = {
  container: {
    width: "100%",
    maxWidth: "600px",
    margin: "0 auto",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
    height: "80vh"
  },
  messagesBox: {
    flex: 1,
    overflowY: "auto",
    border: "1px solid #ddd",
    padding: "10px",
    borderRadius: "8px",
    marginBottom: "12px",
    background: "#fafafa"
  },
  message: {
    padding: "8px 12px",
    borderRadius: "8px",
    marginBottom: "10px",
    maxWidth: "80%",
  },
  userMsg: {
    background: "#007bff",
    color: "white",
    alignSelf: "flex-end",
  },
  botMsg: {
    background: "#eaeaea",
    color: "black",
    alignSelf: "flex-start",
  },
  inputRow: {
    display: "flex",
    gap: "10px",
  },
  input: {
    flex: 1,
    padding: "10px",
    borderRadius: "6px",
    border: "1px solid #ccc",
  },
  button: {
    padding: "10px 14px",
    borderRadius: "6px",
    border: "none",
    background: "#28a745",
    color: "white",
    cursor: "pointer",
  }
};
