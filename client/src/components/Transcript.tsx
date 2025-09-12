import React from "react";
import type { TranscriptLine } from "../hooks/useVoice";

export const Transcript: React.FC<{ lines: TranscriptLine[] }> = ({ lines }) => {
  return (
    <div className="card" style={{ height: 160, overflowY: "auto", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }}>
      {lines.length === 0 && <div style={{ color: "#6b7280" }}>No transcript yet…</div>}
      {lines.map((l, i) => (
        <div key={i} style={{ color: "#111827", fontWeight: 600 }}>
          {l.text}
        </div>
      ))}
    </div>
  );
};
