import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

async function fetchMetrics() {
  const res = await fetch("/metrics");
  return res.json();
}

async function fetchTurns() {
  const res = await fetch("/metrics/turns");
  return res.json();
}

export const MetricsPanel: React.FC = () => {
  const [showTurns, setShowTurns] = useState(false);
  
  const { data, isLoading } = useQuery({
    queryKey: ["metrics"],
    queryFn: fetchMetrics,
    refetchInterval: 2000,
  });

  const { data: turnsData, isLoading: turnsLoading } = useQuery({
    queryKey: ["turns"],
    queryFn: fetchTurns,
    refetchInterval: 2000,
    enabled: showTurns,
  });

  if (isLoading) return <div>Loading metrics…</div>;
  if (!data) return <div>No metrics</div>;

  const m = data.metrics;
  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 12, color: "#6b7280" }}>
        Total Turns: <strong>{data.turns}</strong>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
            <th style={{ textAlign: "left", padding: "8px 0", fontWeight: 600, color: "#374151" }}>Metric</th>
            <th style={{ textAlign: "right", padding: "8px 0", fontWeight: 600, color: "#374151" }}>P50 (ms)</th>
            <th style={{ textAlign: "right", padding: "8px 0", fontWeight: 600, color: "#374151" }}>P95 (ms)</th>
            <th style={{ textAlign: "right", padding: "8px 0", fontWeight: 600, color: "#374151" }}>Count</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(m).map(([k, v]: any) => (
            <tr key={k} style={{ borderBottom: "1px solid #f3f4f6" }}>
              <td style={{ padding: "8px 0", fontWeight: 500, color: "#111827" }}>
                {k.replace('_ms', '').toUpperCase()}
              </td>
              <td style={{ textAlign: "right", padding: "8px 0", color: "#059669", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
                {v.p50}
              </td>
              <td style={{ textAlign: "right", padding: "8px 0", color: "#dc2626", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
                {v.p95}
              </td>
              <td style={{ textAlign: "right", padding: "8px 0", color: "#6b7280", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
                {v.count}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Individual Turns Section */}
      <div style={{ marginTop: 16, borderTop: "1px solid #e5e7eb", paddingTop: 12 }}>
        <button
          onClick={() => setShowTurns(!showTurns)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12,
            fontWeight: 600,
            color: "#374151",
            padding: 0,
            marginBottom: 8
          }}
        >
          <span>{showTurns ? "▼" : "▶"}</span>
          Individual Turns ({data.turns})
        </button>
        
        {showTurns && (
          <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: 6 }}>
            {turnsLoading ? (
              <div style={{ padding: 12, textAlign: "center", color: "#6b7280" }}>Loading turns...</div>
            ) : turnsData && turnsData.length > 0 ? (
              <div>
                {turnsData.map((turn: any, index: number) => (
                  <div key={index} style={{ 
                    padding: "8px 12px", 
                    borderBottom: "1px solid #f3f4f6",
                    fontSize: 11,
                    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace"
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <span style={{ fontWeight: 600, color: "#111827" }}>
                        Turn #{turn.turn} {turn.interrupted ? "(Interrupted)" : ""}
                      </span>
                      <span style={{ color: "#6b7280", fontSize: 10 }}>
                        {new Date(turn.t * 1000).toLocaleTimeString()}
                      </span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, color: "#6b7280" }}>
                      <div>RT: <span style={{ color: "#059669" }}>{turn.rt_ms}ms</span></div>
                      <div>ASR: <span style={{ color: "#059669" }}>{turn.asr_ms}ms</span></div>
                      <div>Plan: <span style={{ color: "#059669" }}>{turn.plan_ms}ms</span></div>
                      <div>TTS: <span style={{ color: "#059669" }}>{turn.tts_ms}ms</span></div>
                    </div>
                    {turn.reason && (
                      <div style={{ marginTop: 4, fontSize: 10, color: "#9ca3af" }}>
                        Reason: {turn.reason}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: 12, textAlign: "center", color: "#6b7280" }}>No turn data available</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
