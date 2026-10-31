import json

import pytest

from neurokiln.store import ExperimentStore, SCHEMA_VERSION


def _arena_rows():
    return [
        {
            "pipeline": "csp_lda",
            "n_splits": 2,
            "balanced_accuracy_mean": 0.8,
            "balanced_accuracy_std": 0.05,
            "cohen_kappa_mean": 0.6,
            "splits": [
                {
                    "split": {
                        "protocol": "within_subject", "fold": 0, "repeat": 0, "seed": 0,
                        "n_train": 8, "n_test": 2, "signature": "abc0123456789def",
                    },
                    "balanced_accuracy": 0.75,
                    "cohen_kappa": 0.5,
                    "duration_ms": 12.5,
                    "y_true": [0, 1],
                    "y_pred": [0, 1],
                },
                {
                    "split": {
                        "protocol": "within_subject", "fold": 1, "repeat": 0, "seed": 0,
                        "n_train": 8, "n_test": 2, "signature": "def0123456789abc",
                    },
                    "balanced_accuracy": 0.85,
                    "cohen_kappa": 0.7,
                    "duration_ms": 11.0,
                    "y_true": [1, 0],
                    "y_pred": [1, 1],
                },
            ],
        },
        {
            "pipeline": "raw_centroid",
            "n_splits": 2,
            "balanced_accuracy_mean": 0.55,
            "balanced_accuracy_std": 0.02,
            "cohen_kappa_mean": 0.1,
            "splits": [
                {
                    "split": {
                        "protocol": "within_subject", "fold": 0, "repeat": 0, "seed": 0,
                        "n_train": 8, "n_test": 2, "signature": "jkl0123456789mno",
                    },
                    "balanced_accuracy": 0.5,
                    "cohen_kappa": 0.0,
                    "duration_ms": 1.0,
                    "y_true": [0, 1],
                    "y_pred": [1, 0],
                },
                {
                    "split": {
                        "protocol": "within_subject", "fold": 1, "repeat": 0, "seed": 0,
                        "n_train": 8, "n_test": 2, "signature": "pqr0123456789stu",
                    },
                    "balanced_accuracy": 0.6,
                    "cohen_kappa": 0.2,
                    "duration_ms": 1.0,
                    "y_true": [1, 0],
                    "y_pred": [1, 1],
                },
            ],
        },
    ]


def _record_kwargs():
    return dict(
        name="demo-run",
        dataset_id="synthetic",
        dataset_name="合成演示数据",
        dataset_version="1",
        dataset_signature="a" * 64,
        dataset_manifest={"dataset": "synthetic", "n_trials": 10},
        protocol="within_subject",
        protocol_params={"folds": 5, "seeds": [0]},
        pipelines=[
            {"id": "csp_lda", "name": "CSP + LDA"},
            {"id": "raw_centroid", "name": "原始均值特征 + 最近质心"},
        ],
        seeds=[0],
        arena_rows=_arena_rows(),
        duration_ms=25.5,
        notes="test",
    )


def test_store_roundtrip(tmp_path):
    db = tmp_path / "neurokiln.db"
    store = ExperimentStore(db)
    store.save_dataset(
        "synthetic", "合成演示数据", "builtin", "a" * 64,
        {"dataset": "synthetic", "n_trials": 10},
    )
    experiment_id = store.record_experiment(**_record_kwargs())
    assert experiment_id == 1

    items, total = store.list_experiments()
    assert total == 1
    assert items[0]["id"] == experiment_id
    assert items[0]["n_runs"] == 4
    assert items[0]["best"]["pipeline"] == "csp_lda"
    assert items[0]["best"]["balanced_accuracy"] == 0.8

    detail = store.get_experiment(experiment_id)
    assert detail is not None
    assert detail["dataset"]["signature"] == "a" * 64
    assert detail["summary"]["best_pipeline"] == "csp_lda"
    assert len(detail["rows"]) == 2
    csp = next(row for row in detail["rows"] if row["pipeline"] == "csp_lda")
    assert len(csp["splits"]) == 2
    assert csp["splits"][0]["y_true"] == [0, 1]
    assert csp["splits"][0]["split"]["signature"] == "abc0123456789def"
    assert csp["splits"][0]["duration_ms"] == 12.5

    stats = store.stats()
    assert stats["experiments"] == 1
    assert stats["completed"] == 1
    assert stats["runs"] == 4
    assert stats["datasets_used"] == 1
    assert stats["best_run"]["pipeline"] == "csp_lda"
    # 最佳结果用折均值，不能被单折最高分抬高
    assert stats["best_run"]["balanced_accuracy"] == 0.8
    assert stats["best_run"]["pipeline_name"] == "CSP + LDA"

    datasets = store.datasets()
    assert datasets[0]["id"] == "synthetic"
    assert datasets[0]["manifest"]["n_trials"] == 10


def test_store_persists_between_instances(tmp_path):
    db = tmp_path / "neurokiln.db"
    first = ExperimentStore(db)
    experiment_id = first.record_experiment(**_record_kwargs())

    second = ExperimentStore(db)
    detail = second.get_experiment(experiment_id)
    assert detail is not None
    assert detail["name"] == "demo-run"
    assert detail["n_runs"] == 4


def test_store_delete_cascades_runs(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    experiment_id = store.record_experiment(**_record_kwargs())
    assert store.delete_experiment(experiment_id) is True
    assert store.get_experiment(experiment_id) is None
    assert store.stats()["runs"] == 0
    assert store.delete_experiment(experiment_id) is False


def test_store_best_run_uses_fold_mean(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    store.record_experiment(**_record_kwargs())

    rows = _arena_rows()
    rows[0]["balanced_accuracy_mean"] = 0.6
    rows[0]["splits"][0]["balanced_accuracy"] = 1.0
    rows[0]["splits"][1]["balanced_accuracy"] = 0.2
    weaker = _record_kwargs()
    weaker.update(name="single-fold-spike", arena_rows=rows)
    store.record_experiment(**weaker)

    best = store.stats()["best_run"]
    assert best["pipeline"] == "csp_lda"
    assert best["balanced_accuracy"] == 0.8
    assert best["name"] == "demo-run"


def test_store_schema_version_and_filters(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    store.record_experiment(**_record_kwargs())
    other = _record_kwargs()
    other.update(name="other", dataset_id="bciciv2a", dataset_name="BCI IV 2a")
    store.record_experiment(**other)

    items, total = store.list_experiments(dataset_id="bciciv2a")
    assert total == 1
    assert items[0]["name"] == "other"

    with store._connect() as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
    assert int(version) == SCHEMA_VERSION


def test_store_experiment_is_idempotent_per_job(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    kwargs = _record_kwargs()
    kwargs["job_id"] = "job-arena-1"
    first = store.record_experiment(**kwargs)
    second = store.record_experiment(**kwargs)
    assert first == second
    items, total = store.list_experiments()
    assert total == 1
    assert items[0]["job_id"] == "job-arena-1"


def test_explanation_job_reuse_restart_and_filters(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    explanation_id = store.create_explanation(
        pipeline="csp_lda",
        pipeline_name="CSP + LDA",
        folder=str(tmp_path / "explanation"),
        job_id="job-xai-1",
    )
    item = store.get_explanation_by_job_id("job-xai-1")
    assert item is not None
    assert item["id"] == explanation_id
    assert item["status"] == "running"

    store.fail_explanation(explanation_id, "boom")
    store.restart_explanation(explanation_id)
    item = store.get_explanation(explanation_id)
    assert item["status"] == "running"
    assert item["error"] is None

    store.finish_explanation(
        explanation_id,
        folder=str(tmp_path / "explanation"),
        summary={"deletion_auc": 0.5},
        warnings=["w"],
        duration_ms=10.0,
    )
    items, total = store.list_explanations(
        pipeline="csp_lda", status="completed"
    )
    assert total == 1
    assert items[0]["summary"]["deletion_auc"] == 0.5

    assert store.fail_explanations_for_jobs(["job-xai-1"], "orphan") == 0


def test_store_mark_failed(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    experiment_id = store.record_experiment(**_record_kwargs())
    store.mark_failed(experiment_id, "boom")
    detail = store.get_experiment(experiment_id)
    assert detail is not None
    assert detail["status"] == "failed"
    assert detail["notes"] == "boom"
    assert store.stats()["failed"] == 1

    items, _ = store.list_experiments(status="completed")
    assert items == []


def test_custom_pipeline_crud(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    spec = {
        "nodes": [
            {"id": "n1", "type": "feature.channel_mean", "params": {}},
            {"id": "n2", "type": "decoder.lda", "params": {"shrinkage": "auto"}},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    pipeline_id = store.save_custom_pipeline("我的 DAG", "说明", spec)
    item = store.get_custom_pipeline(pipeline_id)
    assert item is not None
    assert item["name"] == "我的 DAG"
    assert item["spec"]["edges"][0]["target"] == "n2"
    assert store.stats()["custom_pipelines"] == 1

    updated = store.save_custom_pipeline("新名字", "更新", spec, pipeline_id=pipeline_id)
    assert updated == pipeline_id
    assert store.get_custom_pipeline(pipeline_id)["name"] == "新名字"
    assert len(store.list_custom_pipelines()) == 1

    assert store.delete_custom_pipeline(pipeline_id) is True
    assert store.get_custom_pipeline(pipeline_id) is None
    assert store.delete_custom_pipeline(pipeline_id) is False


def test_custom_pipeline_search_space_roundtrip(tmp_path):
    store = ExperimentStore(tmp_path / "neurokiln.db")
    pipeline_id = store.save_custom_pipeline("可调丹方", "", {
        "nodes": [
            {"id": "bp", "type": "preprocess.bandpass", "params": {"low": 8, "high": 30}},
            {"id": "mean", "type": "feature.channel_mean", "params": {}},
            {"id": "lda", "type": "decoder.lda", "params": {}},
        ],
        "edges": [
            {"source": "bp", "target": "mean"},
            {"source": "mean", "target": "lda"},
        ],
    })
    dims = [
        {"node_id": "bp", "param": "high", "kind": "float",
         "low": 20.0, "high": 40.0, "step": 0.5, "default": 30.0},
    ]
    assert store.get_custom_pipeline_space(pipeline_id) is None
    store.save_custom_pipeline_space(pipeline_id, dims)
    assert store.get_custom_pipeline_space(pipeline_id) == dims
    assert store.list_custom_pipeline_spaces() == {pipeline_id: dims}
    store.save_custom_pipeline_space(pipeline_id, dims[:1])
    assert store.get_custom_pipeline_space(pipeline_id) == dims[:1]

    assert store.delete_custom_pipeline(pipeline_id) is True
    assert store.get_custom_pipeline_space(pipeline_id) is None
    assert store.list_custom_pipeline_spaces() == {}
    with pytest.raises(KeyError):
        store.save_custom_pipeline_space(pipeline_id, dims)
