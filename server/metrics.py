from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import orjson

from .settings import settings


@dataclass
class Percentiles:
    count: int
    p50: int
    p95: int


def _percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    if p <= 0:
        return int(sorted_vals[0])
    if p >= 1:
        return int(sorted_vals[-1])
    k = p * (len(sorted_vals) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return int(sorted_vals[f])
    # linear interpolation
    return int(round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 0))


def _summarize(vals: List[int]) -> Percentiles:
    vals_sorted = sorted(int(v) for v in vals if v is not None)
    return Percentiles(
        count=len(vals_sorted),
        p50=_percentile(vals_sorted, 0.50),
        p95=_percentile(vals_sorted, 0.95),
    )


def read_turn_metrics(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                if obj.get("evt") == "turn_metrics":
                    out.append(obj)
    except Exception:
        return []
    return out


def summarize_turns(turns: List[Dict]) -> Dict[str, Dict]:
    # Collect each metric
    keys = ("rt_ms", "asr_ms", "plan_ms", "tts_ms")
    buckets: Dict[str, List[int]] = {k: [] for k in keys}
    for t in turns:
        for k in keys:
            v = t.get(k)
            if isinstance(v, (int, float)):
                buckets[k].append(int(v))

    return {
        k: {
            "count": s.count,
            "p50": s.p50,
            "p95": s.p95,
        }
        for k, s in ((k, _summarize(vs)) for k, vs in buckets.items())
    }


def summarize_file(path: str | Path | None = None) -> Dict:
    path = path or settings.metrics_file
    turns = read_turn_metrics(path)
    return {
        "turns": len(turns),
        "metrics": summarize_turns(turns),
    }
