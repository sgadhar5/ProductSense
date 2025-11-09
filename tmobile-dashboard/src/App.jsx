import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import "./styles/layout.css";

import FixedLogo from "./components/FixedLogo";
import "./styles/fixedlogo.css";

import Dashboard from "./pages/Dashboard";
import Backlog from "./pages/Backlog";
import Chat from "./pages/Chatbot";

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-100">

        <div className="flex flex-col flex-1 overflow-hidden">
          <Header />

          <main className="p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/backlog" element={<Backlog />} />
              <Route path="/chat" element={<Chat />} />
            </Routes>
            <FixedLogo />
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
