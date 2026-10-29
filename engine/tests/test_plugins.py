import io
import json
import zipfile
from pathlib import Path

import pytest

from neurokiln.bundle import export_bundle, reproduce_bundle
from neurokiln.data import DatasetRegistry, make_synthetic_epochs
from neurokiln.evaluate import run_arena
from neurokiln.graph import node_catalog
from neurokiln.plugins import (
    PluginError,
    PluginManager,
    normalize_manifest,
    plugin_compatibility,
    validate_plugin_source,
)
from neurokiln.protocol import WithinSubjectProtocol
from neurokiln.runner import ExperimentRequest, resolve_pipelines, run_experiment
from neurokiln.store import ExperimentStore


def _graph_manifest(plugin_id="zip_demo"):
    return {
        "id": plugin_id,
        "name": "ZIP 图插件",
        "version": "1.0.0",
        "description": "测试用声明式插件",
        "runtime": "graph",
        "permissions": ["graph"],
        "pipelines": [
            {
                "id": "var_lda",
                "name": "方差 + LDA",
                "paradigm": "MI",
                "graph": {
                    "nodes": [
                        {"id": "var", "type": "feature.channel_std", "params": {}},
                        {"id": "scale", "type": "feature.standardize", "params": {}},
                        {"id": "clf", "type": "decoder.lda", "params": {}},
                    ],
                    "edges": [
                        {"source": "var", "target": "scale"},
                        {"source": "scale", "target": "clf"},
                    ],
                },
            }
        ],
    }


def _v2_manifest(plugin_id="zip_demo", *, engine_min_version="0.1.0"):
    manifest = _graph_manifest(plugin_id)
    manifest.update(
        {
            "schema_version": 2,
            "engine_min_version": engine_min_version,
            "api_version": 1,
        }
    )
    return manifest


def _zip_bytes(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            archive.writestr(name, payload)
    return buffer.getvalue()


def _python_plugin_files(plugin_id="demo_nodes"):
    manifest = {
        "id": plugin_id,
        "name": "Python 节点插件",
        "version": "0.1.0",
        "description": "测试受限模式与动态节点",
        "runtime": "python",
        "permissions": ["nodes"],
        "entry": "plugin.py",
        "pipelines": [
            {
                "id": "var_lda",
                "name": "插件方差 + LDA",
                "paradigm": "MI",
                "graph": {
                    "nodes": [
                        {"id": "var", "type": f"plugin.{plugin_id}.plugin_var", "params": {}},
                        {"id": "scale", "type": "feature.standardize", "params": {}},
                        {"id": "clf", "type": "decoder.lda", "params": {}},
                    ],
                    "edges": [
                        {"source": "var", "target": "scale"},
                        {"source": "scale", "target": "clf"},
                    ],
                },
            }
        ],
    }
    code = """
import numpy as np
from neurokiln.pipeline.nodes import Node


class PluginVar(Node):
    name = "plugin_var"
    kind = "feature"

    def transform(self, X):
        return np.asarray(X).std(axis=2)


def register(api):
    api.register_node({
        "type": "plugin.demo_nodes.plugin_var",
        "label": "插件方差特征",
        "kind": "feature",
        "description": "测试用插件节点",
        "params": [],
        "factory": PluginVar,
    })
""".replace("plugin.demo_nodes.", f"plugin.{plugin_id}.")
    return {"plugin.json": json.dumps(manifest), "plugin.py": code}


def test_bundled_graph_plugin_lifecycle(tmp_path):
    manager = PluginManager(tmp_path / "plugins", trusted=False)
    catalog_ids = {item["id"] for item in manager.catalog()}
    assert {"mu_beta_erd", "p300_compact", "ssvep_fbcca_wide"} <= catalog_ids

    item = manager.install_bundled("mu_beta_erd")
    assert item["enabled"] and item["active"] and item["verified"]
    identifier = "plugin:mu_beta_erd/mu_variance_lda"
    definitions = manager.definitions()
    assert definitions[0]["id"] == identifier
    assert definitions[0]["plugin"]["package_sha256"]
    assert definitions[0]["plugin"]["manifest_sha256"]
    tampered = {
        **definitions[0],
        "plugin": {**definitions[0]["plugin"], "manifest_sha256": "0" * 64},
    }
    with pytest.raises(PluginError, match="清单哈希"):
        manager.require_definition(tampered)

    epochs = make_synthetic_epochs(n_trials_per_class=10, seed=2)
    splits = WithinSubjectProtocol(folds=3, seeds=(1,)).split(epochs)
    _, pipelines = resolve_pipelines([identifier], plugins=manager)
    rows = run_arena(epochs, pipelines, splits)
    assert rows[0]["pipeline"] == identifier

    manager.disable("mu_beta_erd")
    assert manager.definitions() == []
    with pytest.raises(PluginError, match="不可用"):
        manager.definition(identifier)
    manager.enable("mu_beta_erd")
    assert manager.definition(identifier)["graph"]["decoder"] == "clf"

    reloaded = PluginManager(tmp_path / "plugins", trusted=False)
    assert reloaded.definition(identifier)["name"] == "μ 节律方差 + LDA"
    reloaded.uninstall("mu_beta_erd")
    assert all(item["installed"] is False for item in reloaded.catalog() if item["id"] == "mu_beta_erd")


def test_zip_install_rejects_traversal_and_verifies_hashes(tmp_path):
    manager = PluginManager(tmp_path / "plugins", trusted=False)
    with pytest.raises(PluginError):
        manager.install_zip(
            _zip_bytes({"plugin.json": json.dumps(_graph_manifest()), "../evil.py": "x = 1"}),
            filename="bad.zip",
        )

    payload = _zip_bytes(
        {"wrapper/plugin.json": json.dumps(_graph_manifest()), "wrapper/notes.txt": "hello"}
    )
    item = manager.install_zip(payload, filename="good.zip")
    assert item["enabled"] and item["verified"]
    installed = tmp_path / "plugins" / "zip_demo"
    (installed / "notes.txt").write_text("tampered", encoding="utf-8")
    manager.disable("zip_demo")
    with pytest.raises(PluginError, match="哈希"):
        manager.enable("zip_demo")
    assert manager.list_installed()[0]["enabled"] is False

    (installed / "notes.txt").write_text("hello", encoding="utf-8")
    manager.enable("zip_demo")
    (installed / "extra.py").write_text("x = 1", encoding="utf-8")
    manager.disable("zip_demo")
    with pytest.raises(PluginError, match="不一致"):
        manager.enable("zip_demo")


def test_python_plugin_requires_trusted_mode(tmp_path):
    files = _python_plugin_files()
    payload = _zip_bytes(files)

    restricted = PluginManager(tmp_path / "restricted", trusted=False)
    with pytest.raises(PluginError, match="受限模式"):
        restricted.install_zip(payload)
    assert restricted.list_installed() == []

    trusted = PluginManager(tmp_path / "trusted", trusted=True)
    item = trusted.install_zip(payload)
    assert item["runtime"] == "python" and item["enabled"]
    assert "plugin.demo_nodes.plugin_var" in {spec["type"] for spec in node_catalog()}
    identifier = "plugin:demo_nodes/var_lda"
    epochs = make_synthetic_epochs(n_trials_per_class=10, seed=4)
    split = WithinSubjectProtocol(folds=3, seeds=(1,)).split(epochs)[0]
    _, pipelines = resolve_pipelines([identifier], plugins=trusted)
    result = pipelines[0].run_split(epochs, split)
    assert result["y_true"].shape == result["y_pred"].shape

    trusted.disable("demo_nodes")
    assert "plugin.demo_nodes.plugin_var" not in {spec["type"] for spec in node_catalog()}

    trusted.enable("demo_nodes")
    (tmp_path / "trusted" / "demo_nodes" / "plugin.py").write_text("broken", encoding="utf-8")
    trusted.disable("demo_nodes")
    with pytest.raises(PluginError, match="哈希"):
        trusted.enable("demo_nodes")


def test_bundle_reproduction_requires_enabled_plugin(tmp_path):
    registry = DatasetRegistry(tmp_path / "data")
    store = ExperimentStore(tmp_path / "neurokiln.db")
    manager = PluginManager(tmp_path / "plugins", trusted=False)
    manager.install_bundled("mu_beta_erd")
    request = ExperimentRequest(
        dataset_id="synthetic",
        subject_ids=["S01", "S02"],
        options={"n_trials_per_class": 10, "n_times": 40},
        protocol="loso",
        folds=2,
        seeds=[1],
        pipelines=["plugin:mu_beta_erd/mu_variance_lda"],
        name="plugin bundle",
    )
    result = run_experiment(registry, store, request, plugins=manager)
    assert result["rows"][0]["pipeline"].startswith("plugin:")
    bundle = export_bundle(store, result["experiment_id"], tmp_path / "bundle")

    with pytest.raises(ValueError, match="插件"):
        reproduce_bundle(registry, bundle["directory"])
    reproduced = reproduce_bundle(registry, bundle["directory"], plugins=manager)
    assert reproduced["matched"]
    assert all(item["difference"] == pytest.approx(0.0) for item in reproduced["comparisons"])

    manager.disable("mu_beta_erd")
    with pytest.raises(ValueError, match="启用"):
        reproduce_bundle(registry, bundle["directory"], plugins=manager)


def test_manifest_v2_compatibility_and_legacy_digest(tmp_path):
    legacy = normalize_manifest(_graph_manifest())
    assert "schema_version" not in legacy
    assert "engine_min_version" not in legacy
    assert plugin_compatibility(legacy) == (True, "旧版清单 v1（兼容模式）")

    missing_engine = _graph_manifest()
    missing_engine["schema_version"] = 2
    with pytest.raises(PluginError, match="engine_min_version"):
        normalize_manifest(missing_engine)

    missing_api = _v2_manifest()
    missing_api.pop("api_version")
    with pytest.raises(PluginError, match="api_version"):
        normalize_manifest(missing_api)

    manager = PluginManager(tmp_path / "plugins", trusted=False)
    with pytest.raises(PluginError, match="不兼容"):
        manager.install_zip(
            _zip_bytes(
                {"plugin.json": json.dumps(_v2_manifest(engine_min_version="99.0.0"))}
            )
        )
    assert manager.list_installed() == []

    item = manager.install_zip(_zip_bytes({"plugin.json": json.dumps(_v2_manifest())}))
    assert item["schema_version"] == 2
    assert item["engine_min_version"] == "0.1.0"
    assert item["api_version"] == 1
    assert item["compatible"] and item["compatibility"] == "兼容"

    status = manager.status()
    assert status["schema_version"] == 2
    assert status["supported_schema_versions"] == [1, 2]
    assert status["node_api_version"] == 1
    assert status["catalog"]
    assert all(entry["schema_version"] == 2 for entry in status["catalog"])
    assert all(entry["compatible"] for entry in status["catalog"])
    definition = manager.definitions()[0]
    assert definition["plugin"]["schema_version"] == 2
    assert definition["plugin"]["api_version"] == 1

    legacy_manager = PluginManager(tmp_path / "legacy", trusted=False)
    legacy_item = legacy_manager.install_zip(
        _zip_bytes({"plugin.json": json.dumps(_graph_manifest("legacy_demo"))})
    )
    assert legacy_item["schema_version"] == 1
    assert legacy_item["compatible"]
    assert legacy_item["enabled"]


def test_validate_plugin_source_checks_package_and_graphs(tmp_path):
    catalog = Path(__file__).resolve().parents[1] / "neurokiln" / "plugin_catalog"
    bundled = validate_plugin_source(catalog / "mu_beta_erd")
    assert bundled["valid"] and bundled["kind"] == "directory"
    assert bundled["manifest"]["schema_version"] == 2
    assert bundled["pipelines"][0]["status"] == "ok"
    assert any("未封存" in warning for warning in bundled["warnings"])

    manager = PluginManager(tmp_path / "plugins", trusted=False)
    manager.install_bundled("mu_beta_erd")
    installed = validate_plugin_source(tmp_path / "plugins" / "mu_beta_erd")
    assert installed["valid"]
    assert installed["package"]["declared"] and installed["package"]["verified"]
    assert installed["package"]["manifest_sha256"] == "verified"

    manager.install_zip(
        _zip_bytes({"plugin.json": json.dumps(_v2_manifest()), "notes.txt": "hello"})
    )
    target = tmp_path / "plugins" / "zip_demo"
    (target / "notes.txt").write_text("tampered", encoding="utf-8")
    tampered = validate_plugin_source(target)
    assert not tampered["valid"]
    assert tampered["package"]["mismatched"] == ["notes.txt"]
    assert any("哈希" in error for error in tampered["errors"])

    broken = _v2_manifest("broken_graph")
    broken["pipelines"][0]["graph"]["nodes"][0]["type"] = "feature.unknown_node"
    broken_path = tmp_path / "broken.zip"
    broken_path.write_bytes(_zip_bytes({"plugin.json": json.dumps(broken)}))
    broken_report = validate_plugin_source(broken_path)
    assert not broken_report["valid"]
    assert broken_report["pipelines"][0]["status"] == "error"

    incompatible = _v2_manifest("future_plugin", engine_min_version="99.0.0")
    incompatible_path = tmp_path / "future.zip"
    incompatible_path.write_bytes(_zip_bytes({"plugin.json": json.dumps(incompatible)}))
    future_report = validate_plugin_source(incompatible_path)
    assert not future_report["valid"]
    assert future_report["compatible"] is False
    assert any("不兼容" in error for error in future_report["errors"])

    python_files = _python_plugin_files("static_demo")
    python_manifest = json.loads(python_files["plugin.json"])
    python_manifest.update(
        {"schema_version": 2, "engine_min_version": "0.1.0", "api_version": 1}
    )
    python_files["plugin.json"] = json.dumps(python_manifest)
    python_path = tmp_path / "python.zip"
    python_path.write_bytes(_zip_bytes(python_files))
    python_report = validate_plugin_source(python_path)
    assert python_report["valid"]
    assert python_report["pipelines"][0]["status"] == "deferred"
    assert any("沙箱" in warning for warning in python_report["warnings"])


def test_cli_plugin_validate(tmp_path, capsys):
    from neurokiln import cli

    catalog = Path(__file__).resolve().parents[1] / "neurokiln" / "plugin_catalog"
    exit_code = cli.main(["plugin", "validate", str(catalog / "mu_beta_erd"), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] and payload["strict"] is False
    assert payload["manifest"]["id"] == "mu_beta_erd"

    exit_code = cli.main(
        ["plugin", "validate", str(catalog / "mu_beta_erd"), "--strict"]
    )
    assert exit_code == 1
    capsys.readouterr()

    broken = _v2_manifest("broken_cli")
    broken["pipelines"][0]["id"] = ""
    broken_file = tmp_path / "broken.json"
    broken_file.write_text(json.dumps(broken), encoding="utf-8")
    exit_code = cli.main(["plugin", "validate", str(broken_file), "--json"])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False


def test_node_catalog_exposes_params_schema_and_io(tmp_path):
    catalog = {spec["type"]: spec for spec in node_catalog()}
    notch = catalog["preprocess.notch"]
    assert notch["params_schema"]["type"] == "object"
    assert notch["params_schema"]["properties"]["freq"]["type"] == "number"
    assert notch["inputs"] == ["epochs"] and notch["outputs"] == ["epochs"]
    assert notch["paradigm"] == "ANY"
    assert notch["requires_fit"] and notch["cacheable"] and notch["deterministic"]
    assert notch["plugin"] is False
    select_k = catalog["feature.select_k"]
    assert select_k["params_schema"]["properties"]["method"]["enum"] == [
        "anova", "mutual_info",
    ]
    assert select_k["inputs"] == ["features"]
    assert select_k["outputs"] == ["features"]

    trusted = PluginManager(tmp_path / "trusted", trusted=True)
    trusted.install_zip(_zip_bytes(_python_plugin_files()))
    plugin = {spec["type"]: spec for spec in node_catalog()}["plugin.demo_nodes.plugin_var"]
    assert plugin["plugin"] is True
    assert plugin["params_schema"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert plugin["inputs"] == ["epochs", "features"]
    assert plugin["outputs"] == ["features"]
    assert plugin["paradigm"] == "ANY"
    assert plugin["requires_fit"] is False
    assert plugin["cacheable"] is False
    trusted.disable("demo_nodes")
