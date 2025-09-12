import { useEffect, useRef, useState } from "react";

export type MetricSummary = {
  turns: number;
  p50: { rt_ms: number; asr_ms: number; plan_ms: number; tts_ms: number };
  p95: { rt_ms: number; asr_ms: number; plan_ms: number; tts_ms: number };
};

export function useMetrics(pollMs = 1000) {
  const [data, setData] = useState<MetricSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    const fetchOnce = async () => {
      try {
        const res = await fetch("/metrics", { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const obj = (await res.json()) as MetricSummary;
        setData(obj);
        setError(null);
      } catch (e: any) {
        setError(e?.message ?? "failed to load metrics");
      }
    };

    fetchOnce();
    timer.current = window.setInterval(fetchOnce, pollMs);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [pollMs]);

  return { data, error };
}
