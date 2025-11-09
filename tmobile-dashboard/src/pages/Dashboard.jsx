import { useEffect, useState } from "react";
import ExecutiveSummary from "../components/ExecutiveSummary";
import SourceVolumes from "../components/SourceVolumes";
import TrendingIssues from "../components/TrendingIssues";
import NegativeStrip from "../components/NegativeStrip";
import RecentTabs from "../components/RecentTabs";
import "../styles/dashboard.css";   // ← new hardcoded CSS

export default function Dashboard() {
  return (
    <div className="dashboard-page-container">
      <div className="dashboard-box">

        <ExecutiveSummary />
        <SourceVolumes />
        <NegativeStrip />

        <div className="dashboard-two-col">
          <div className="left-col">
            <TrendingIssues />
          </div>
          <div className="right-col">
            <RecentTabs />
          </div>
        </div>

      </div>
    </div>
  );
}

