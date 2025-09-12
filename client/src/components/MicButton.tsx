import React from "react";

type Props = { connected: boolean; onStart: () => void; onStop: () => void; onBarge: () => void; };

export const MicButton: React.FC<Props> = ({ connected, onStart, onStop, onBarge }) => {
  return (
    <div className="row" style={{ gap: 8 }}>
      <button className="btn primary" onClick={connected ? onStop : onStart}>
        {connected ? "Stop" : "Start"}
      </button>
      <button className="btn" onClick={onBarge} disabled={!connected}>
        Barge-in
      </button>
    </div>
  );
};
