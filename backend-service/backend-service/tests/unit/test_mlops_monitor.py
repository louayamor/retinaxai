from __future__ import annotations

from app.observability.mlops_monitor import build_monitor_snapshot


def test_build_monitor_snapshot_shapes_payload() -> None:
    generated_at = "2026-05-18T10:30:00+00:00"
    metrics = {
        "imaging": {"accuracy": 0.9},
        "clinical": None,
        "training_summary": {"best_epoch": 3},
    }
    prometheus = {"gpu_utilization": 50.0}

    snapshot = build_monitor_snapshot(
        metrics=metrics,
        prometheus=prometheus,
        generated_at=generated_at,
    )

    assert snapshot["generated_at"] == generated_at
    assert snapshot["metrics"] == {
        "imaging": {"accuracy": 0.9},
        "clinical": None,
    }
    assert snapshot["training_summary"] == {"best_epoch": 3}
    assert snapshot["prometheus"] == {"gpu_utilization": 50.0}


def test_build_monitor_snapshot_defaults_missing_data() -> None:
    snapshot = build_monitor_snapshot(metrics=None, prometheus=None, generated_at=None)

    assert snapshot["metrics"] == {"imaging": None, "clinical": None}
    assert snapshot["training_summary"] is None
    assert snapshot["prometheus"] == {}
