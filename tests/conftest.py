import os
import pytest

@pytest.fixture(autouse=True)
def _test_env(tmp_path, monkeypatch):
    # Ensure tests don’t write to repo root
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("VHYS_METRICS_DIR", str(metrics_dir))
    monkeypatch.setenv("VHYS_METRICS_FILE", str(metrics_dir / "latency.ndjson"))
    yield
