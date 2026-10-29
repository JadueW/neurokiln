import json
import zipfile
from pathlib import Path

import pytest

from neurokiln import cli
from neurokiln.bundle import export_bundle, reproduce_bundle
from neurokiln.data import DatasetRegistry
from neurokiln.runner import ExperimentRequest, run_experiment
from neurokiln.store import ExperimentStore


@pytest.fixture()
def workspace(tmp_path: Path):
    registry = DatasetRegistry(tmp_path / "data")
    store = ExperimentStore(tmp_path / "neurokiln.db")
    return registry, store, tmp_path


def _run(workspace, save: bool = True):
    registry, store, tmp_path = workspace
    request = ExperimentRequest(
        dataset_id="synthetic",
        subject_ids=["S01", "S02"],
        options={"n_trials_per_class": 12, "n_times": 48},
        protocol="loso",
        folds=2,
        seeds=[3],
        pipelines=["csp_lda", "fbcsp_lda"],
        name="runner bundle test",
        save=save,
    )
    return run_experiment(registry, store, request)


def test_run_experiment_records_subject_level_statistics(workspace):
    result = _run(workspace)
    assert result["saved"]
    assert result["experiment_id"] == 1
    assert result["statistics"]["unit"] == "subject"
    assert result["statistics"]["n_units"] == 2
    assert result["statistics"]["ci"]["pipelines"]["csp_lda"]["ci95"][0] is not None
    detail = workspace[1].get_experiment(1)
    assert detail["rows"][0]["balanced_accuracy_ci95"] is not None
    assert detail["summary"]["statistics"]["unit"] == "subject"


def test_export_bundle_and_reproduce_match(workspace):
    registry, store, tmp_path = workspace
    _run(workspace)
    bundle = export_bundle(store, 1, tmp_path / "bundle")
    files = set(bundle["files"])
    assert {"experiment.json", "protocol.json", "data_manifest.json", "environment.json"} <= files
    assert {"metrics.csv", "report.html", "report.xlsx", "reproduce.sh"} <= files
    assert bundle["predictions"] in {"predictions.parquet", "predictions.csv"}
    if bundle["predictions"].endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(tmp_path / "bundle" / "predictions.parquet")
        assert table.num_rows > 0
        assert {"subject_id", "trial_id", "pipeline", "y_true", "y_pred"} <= set(table.column_names)
    with zipfile.ZipFile(tmp_path / "bundle" / "report.xlsx") as archive:
        blob = b"".join(
            archive.read(name) for name in archive.namelist() if name.endswith(".xml")
        )
    assert "95% CI".encode("utf-8") in blob

    reproduced = reproduce_bundle(registry, tmp_path / "bundle")
    assert reproduced["matched"]
    assert reproduced["data_signature"] == reproduced["expected_signature"]
    assert all(item["difference"] == pytest.approx(0.0) for item in reproduced["comparisons"])


def test_bundle_rejects_signature_mismatch(workspace):
    registry, store, tmp_path = workspace
    result = _run(workspace)
    bundle = export_bundle(store, 1, tmp_path / "bundle")
    config_path = Path(bundle["directory"]) / "experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["id"] = "synthetic"
    config["dataset"]["signature"] = "deadbeef"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="数据签名不匹配"):
        reproduce_bundle(registry, bundle["directory"])
    tolerated = reproduce_bundle(
        registry, bundle["directory"], allow_data_mismatch=True
    )
    assert tolerated["data_signature"] != "deadbeef"
    assert tolerated["matched"] is True
    assert result["statistics"]["unit"] == "subject"


def test_cli_arena_export_and_reproduce(workspace, tmp_path, capsys):
    _, _, _ = workspace
    data_dir = tmp_path / "cli-data"
    db = tmp_path / "cli.db"
    common = ["--data-dir", str(data_dir), "--db", str(db)]
    exit_code = cli.main(
        [
            *common,
            "arena",
            "--dataset", "synthetic",
            "--subjects", "S01,S02",
            "--protocol", "loso",
            "--pipeline", "csp_lda",
            "--pipeline", "fbcsp_lda",
            "--name", "cli arena",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == 1
    assert payload["statistics"]["unit"] == "subject"

    exit_code = cli.main(
        [
            *common,
            "report",
            "--experiment", "1",
            "--out", str(tmp_path / "cli-report.html"),
            "--xlsx", str(tmp_path / "cli-report.xlsx"),
            "--json",
        ]
    )
    assert exit_code == 0
    report_paths = json.loads(capsys.readouterr().out)
    assert Path(report_paths["html"]).stat().st_size > 1000
    assert Path(report_paths["xlsx"]).stat().st_size > 1000

    exit_code = cli.main(
        [*common, "export", "--experiment", "1", "--out", str(tmp_path / "cli-bundle"), "--json"]
    )
    assert exit_code == 0
    bundle = json.loads(capsys.readouterr().out)
    assert Path(bundle["directory"]).is_dir()

    exit_code = cli.main([*common, "reproduce", bundle["directory"], "--json"])
    assert exit_code == 0
    reproduced = json.loads(capsys.readouterr().out)
    assert reproduced["matched"] is True
    assert reproduced["comparisons"]
