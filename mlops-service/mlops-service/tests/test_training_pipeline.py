from __future__ import annotations

from app.training.orchestration.training_pipeline import TrainingPipeline


def test_training_pipeline_runs_imaging_stages(monkeypatch):
    calls: list[str] = []

    def make_stage(name: str):
        def _stage():
            calls.append(name)

        return _stage

    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_ingest",
        make_stage("imaging_ingest"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_clean",
        make_stage("imaging_clean"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_transform",
        make_stage("imaging_transform"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_train",
        make_stage("imaging_train"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_evaluate",
        make_stage("imaging_evaluate"),
    )

    TrainingPipeline().run()

    assert calls == [
        "imaging_ingest",
        "imaging_clean",
        "imaging_transform",
        "imaging_train",
        "imaging_evaluate",
    ]


def test_training_pipeline_run_imaging_only(monkeypatch):
    calls: list[str] = []

    def _stage(name: str):
        def fn():
            calls.append(name)

        return fn

    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_ingest",
        _stage("imaging_ingest"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_clean",
        _stage("imaging_clean"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_transform",
        _stage("imaging_transform"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_train",
        _stage("imaging_train"),
    )
    monkeypatch.setattr(
        "app.training.orchestration.training_pipeline.imaging_evaluate",
        _stage("imaging_evaluate"),
    )

    pipeline = TrainingPipeline()
    pipeline.run_imaging()

    assert calls == [
        "imaging_ingest",
        "imaging_clean",
        "imaging_transform",
        "imaging_train",
        "imaging_evaluate",
    ]
