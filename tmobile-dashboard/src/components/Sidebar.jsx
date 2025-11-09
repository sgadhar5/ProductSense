import { Link, useLocation } from "react-router-dom";
import { LayoutDashboard, ClipboardList, MessageSquare } from "lucide-react";

export default function Sidebar() {
  const { pathname } = useLocation();

  const nav = [
    { label: "Dashboard", to: "/", icon: <LayoutDashboard size={20} /> },
    { label: "Backlog", to: "/backlog", icon: <ClipboardList size={20} /> },
    { label: "AI Assistant", to: "/chat", icon: <MessageSquare size={20} /> },
  ];

  return (
    <aside className="w-64 bg-white border-r shadow-sm flex flex-col">
      <div className="p-6 text-xl font-bold text-purple-700">
        PulseAI
      </div>

      <nav className="flex flex-col space-y-1 px-3">
        {nav.map(item => (
          <Link
            key={item.to}
            to={item.to}
            className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-purple-50 hover:text-purple-700 transition
              ${pathname === item.to ? "bg-purple-100 text-purple-700" : ""}`}
          >
            {item.icon}
            <span className="font-medium">{item.label}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
}
