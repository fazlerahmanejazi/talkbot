from pathlib import Path
import orjson

from server.metrics import read_turn_metrics, summarize_turns, summarize_file


def write_turn(fp, turn, rt, asr, plan, tts, interrupted=False):
    fp.write(
        orjson.dumps(
            {
                "evt": "turn_metrics",
                "sid": "test",
                "turn": turn,
                "rt_ms": rt,
                "asr_ms": asr,
                "plan_ms": plan,
                "tts_ms": tts,
                "interrupted": interrupted,
            }
        ).decode("utf-8")
        + "\n"
    )


def test_metrics_p50_p95(tmp_path, monkeypatch):
    metrics_file = tmp_path / "latency.ndjson"
    # Make the settings module point to our temp file
    monkeypatch.setenv("VHYS_METRICS_FILE", str(metrics_file))
    monkeypatch.setenv("VHYS_METRICS_DIR", str(tmp_path))

    # Write synthetic distribution (skewed a bit)
    with metrics_file.open("w", encoding="utf-8") as f:
        # rt_ms: 420, 480, 510, 530, 590, 610  (p95 should be between 590..610)
        write_turn(f, 1, 420, 180, 60, 150)
        write_turn(f, 2, 480, 190, 70, 160)
        write_turn(f, 3, 510, 200, 80, 170)
        write_turn(f, 4, 530, 185, 65, 155)
        write_turn(f, 5, 590, 210, 90, 180)
        write_turn(f, 6, 610, 220, 95, 190)

    # Low-level read + summarize
    turns = read_turn_metrics(metrics_file)
    assert len(turns) == 6

    summary = summarize_turns(turns)
    assert summary["rt_ms"]["count"] == 6
    assert 480 <= summary["rt_ms"]["p50"] <= 530
    assert 590 <= summary["rt_ms"]["p95"] <= 610

    # Full-file helper
    top = summarize_file(metrics_file)
    assert top["turns"] == 6
    assert "metrics" in top
    assert top["metrics"]["asr_ms"]["p50"] >= 180
    assert top["metrics"]["tts_ms"]["p95"] >= 170

    # Budget check example (your README target)
    assert top["metrics"]["rt_ms"]["p95"] <= 610
