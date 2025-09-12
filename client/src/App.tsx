import React, { useState } from "react";
import { useVoice } from "./hooks/useVoice";
import { MetricsPanel } from "./components/MetricsPanel";

export const App: React.FC = () => {
  const { connected, speaking, framesSent, queuedSec, startMic, stopMic, bargeIn, debugTts } = useVoice();
  const [showMetrics, setShowMetrics] = useState(false);

  const resetMetrics = async () => {
    try {
      const response = await fetch("/metrics", { method: "DELETE" });
      if (response.ok) {
        // Refresh the page to show updated metrics
        window.location.reload();
      } else {
        console.error("Failed to reset metrics");
      }
    } catch (error) {
      console.error("Error resetting metrics:", error);
    }
  };

  return (
    <div style={{ padding: 16, maxWidth: 900, margin: "0 auto", fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <h1 style={{ margin: 0 }}>TalkBot</h1>
        <button 
          onClick={() => setShowMetrics(!showMetrics)}
          style={{ 
            padding: "8px 16px", 
            borderRadius: 8, 
            background: showMetrics ? "#6b7280" : "#f3f4f6", 
            color: showMetrics ? "white" : "#374151", 
            border: "1px solid #d1d5db",
            fontSize: 14,
            cursor: "pointer"
          }}
        >
          {showMetrics ? "Hide Metrics" : "Show Metrics"}
        </button>
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 20 }}>
        <button 
          onClick={connected ? stopMic : startMic} 
          style={{ 
            padding: "12px 24px", 
            borderRadius: 12, 
            background: connected ? "#dc2626" : "#2563eb", 
            color: "white", 
            border: 0,
            fontSize: 16,
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
        >
          {connected ? "Stop Voice" : "Start Voice"}
        </button>
        <button 
          onClick={bargeIn} 
          disabled={!connected} 
          style={{ 
            padding: "12px 24px", 
            borderRadius: 12, 
            background: !connected ? "#f3f4f6" : "#f59e0b", 
            color: !connected ? "#9ca3af" : "white", 
            border: 0,
            fontSize: 16,
            fontWeight: 600,
            cursor: connected ? "pointer" : "not-allowed",
            transition: "all 0.2s ease"
          }}
        >
          Barge In
        </button>
        <button 
          onClick={() => debugTts("Hello from Piper")} 
          style={{ 
            padding: "12px 24px", 
            borderRadius: 12, 
            background: "#10b981", 
            color: "white", 
            border: 0,
            fontSize: 16,
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
        >
          Test Voice
        </button>
      </div>

      <div style={{ 
        display: "flex", 
        gap: 12, 
        marginBottom: 20,
        padding: "12px 16px",
        background: "#f9fafb",
        borderRadius: 12,
        border: "1px solid #e5e7eb"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ 
            width: 8, 
            height: 8, 
            borderRadius: "50%", 
            background: connected ? "#10b981" : "#ef4444" 
          }}></div>
          <span style={{ fontSize: 14, color: "#374151" }}>
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <div style={{ color: "#6b7280" }}>•</div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ 
            width: 8, 
            height: 8, 
            borderRadius: "50%", 
            background: speaking ? "#f59e0b" : "#10b981" 
          }}></div>
          <span style={{ fontSize: 14, color: "#374151" }}>
            {speaking ? "Speaking" : "Listening"}
          </span>
        </div>
      </div>


      {showMetrics && (
        <div style={{ marginTop: 24 }}>
          <div style={{ 
            border: "1px solid #e5e7eb", 
            borderRadius: 16, 
            padding: 20, 
            background: "#ffffff",
            boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)"
          }}>
            <div style={{ 
              display: "flex", 
              justifyContent: "space-between", 
              alignItems: "center", 
              marginBottom: 16 
            }}>
              <div style={{ 
                fontSize: 18, 
                fontWeight: 600, 
                color: "#374151" 
              }}>
                Performance Metrics
              </div>
              <button 
                onClick={resetMetrics}
                style={{ 
                  padding: "8px 16px", 
                  borderRadius: 8, 
                  background: "#dc2626", 
                  color: "white", 
                  border: 0,
                  fontSize: 14,
                  fontWeight: 500,
                  cursor: "pointer",
                  transition: "all 0.2s ease"
                }}
              >
                Reset Metrics
              </button>
            </div>
            <MetricsPanel />
          </div>
        </div>
      )}
    </div>
  );
};
