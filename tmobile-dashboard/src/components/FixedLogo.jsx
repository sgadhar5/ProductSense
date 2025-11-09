import React from "react";
import "../styles/fixedlogo.css";  // or whatever your CSS file is
import logo from "../blackcat.PNG";

export default function FixedLogo() {
  return (
    <div className="fixed-logo-container">
      <img src={logo} className="fixed-logo-img" />
    </div>
  );
}
