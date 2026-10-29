import json

import numpy as np
import pytest

from neurokiln.data import DatasetRegistry
from neurokiln.hpo import (
    HPORequest,
    _candidate_record,
    _decode,
    _failure_summary,
    _validate_candidate,
    custom_template,
    get_template,
    normalize_search_space,
    objective_catalog,
    run_hpo,
    template_catalog,
)


@pytest.fixture()
def registry(tmp_path):
    return DatasetRegistry(tmp_path / "data")


def _request(**overrides):
    payload = dict(
        dataset_id="synthetic",
        subject_ids=["S01", "S02"],
        options={"n_trials_per_class": 10, "n_times": 48},
        protocol="loso",
        folds=2,
        seeds=[0],
        template_id="csp_lda",
        objectives=["balanced_accuracy", "cohen_kappa", "channel_count", "duration_ms"],
        budget=8,
        population=3,
        proxy_folds=1,
        full_candidates=2,
        seed=11,
    )
    payload.update(overrides)
    return HPORequest(**payload)


def test_catalogs_are_json_serializable():
    templates = template_catalog()
    assert {item["id"] for item in templates} >= {
        "csp_lda", "fbcsp_lda", "riemann_mdm", "ts_lr", "eegnet", "shallowconvnet"
    }
    assert all(json.dumps(item, ensure_ascii=False) for item in templates)
    objectives = {item["name"]: item for item in objective_catalog()}
    assert objectives["balanced_accuracy"]["direction"] == "max"
    assert objectives["duration_ms"]["direction"] == "min"


def test_template_builds_channel_select_and_clamps_components():
    template = get_template("csp_lda")
    params = {
        **template.defaults(),
        "channel_ratio": 0.5,
        "n_components": 10,
    }
    spec, effective = template.build(params, 8)
    types = [node["type"] for node in spec["nodes"]]
    assert "preprocess.channel_select" in types
    assert effective["channels_used"] == 4
    assert effective["n_components"] == 4
    csp = next(node for node in spec["nodes"] if node["type"] == "feature.csp")
    assert csp["params"]["n_components"] == 4


def test_hpo_runs_proxy_then_full_and_returns_pareto(registry):
    result = run_hpo(registry, _request())
    assert result["format"] == "neurokiln-hpo/v1"
    search = result["search"]
    assert search["evaluations"] == 8
    assert search["proxy_evaluations"] == 6
    assert search["full_evaluations"] == 2
    assert search["proxy_folds"] == 1
    assert search["full_splits"] == 2
    assert len(result["history"]) == 8
    stages = {item["stage"] for item in result["history"]}
    assert stages == {"proxy", "full"}
    assert result["pareto"]
    assert result["best"]["graph"]
    assert result["best"]["definition"]["graph"] == result["best"]["graph"]
    metrics = result["best"]["metrics"]
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["channel_count"] >= 2
    assert metrics["duration_ms"] > 0
    json.dumps(result, ensure_ascii=False)


def test_hpo_is_deterministic_for_accuracy_objectives(registry):
    first = run_hpo(registry, _request(seed=5, objectives=["balanced_accuracy", "cohen_kappa"]))
    second = run_hpo(registry, _request(seed=5, objectives=["balanced_accuracy", "cohen_kappa"]))
    first_params = [item["params"] for item in first["history"]]
    second_params = [item["params"] for item in second["history"]]
    assert first_params == second_params
    assert first["best"]["params"] == second["best"]["params"]


def test_invalid_requests_are_rejected(registry):
    with pytest.raises(ValueError, match="template"):
        run_hpo(registry, _request(template_id="missing"))
    with pytest.raises(ValueError, match="budget"):
        run_hpo(registry, _request(budget=2))
    with pytest.raises(ValueError, match="population"):
        run_hpo(registry, _request(population=8))
    with pytest.raises(ValueError, match="objective"):
        run_hpo(registry, _request(objectives=["auc"]))


def _custom_spec(channel_select: bool = False):
    nodes = [
        {"id": "bp", "type": "preprocess.bandpass", "params": {"low": 8, "high": 30}},
        {"id": "csp", "type": "feature.csp", "params": {"n_components": 4}},
        {"id": "select", "type": "feature.select_k", "params": {"k": 4}},
        {"id": "lda", "type": "decoder.lda", "params": {"shrinkage": "auto"}},
    ]
    edges = [
        {"source": "bp", "target": "csp"},
        {"source": "csp", "target": "select"},
        {"source": "select", "target": "lda"},
    ]
    if channel_select:
        nodes.insert(1, {"id": "chan", "type": "preprocess.channel_select",
                         "params": {"n_channels": 6}})
        edges = [
            {"source": "bp", "target": "chan"},
            {"source": "chan", "target": "csp"},
            {"source": "csp", "target": "select"},
            {"source": "select", "target": "lda"},
        ]
    return {"nodes": nodes, "edges": edges}


_CUSTOM_DIMS = [
    {"node_id": "bp", "param": "high", "low": 20, "high": 40, "step": 0.5},
    {"node_id": "csp", "param": "n_components", "low": 2, "high": 8, "step": 2},
    {"node_id": "select", "param": "method", "options": ["anova", "mutual_info"]},
]


def test_custom_template_builds_and_overrides_node_params():
    template = custom_template(7, "我的丹方", "逐节点覆盖参数", _custom_spec(), _CUSTOM_DIMS)
    assert template.id == "custom:7"
    assert template.family == "custom"
    assert [item.name for item in template.space] == [
        "bp::high", "csp::n_components", "select::method",
    ]
    params = {**template.defaults(), "bp::high": 35.0, "csp::n_components": 6,
              "select::method": "mutual_info"}
    spec, effective = template.build(params, 8)
    by_id = {node["id"]: node for node in spec["nodes"]}
    assert by_id["bp"]["params"]["high"] == 35.0
    assert by_id["csp"]["params"]["n_components"] == 6
    assert by_id["select"]["params"]["method"] == "mutual_info"
    assert effective["channels_used"] == 8
    assert template.source["spec_hash"]
    assert template.source["space_hash"]


def test_custom_template_reports_channels_used():
    template = custom_template(8, "通道搜索", "", _custom_spec(channel_select=True), _CUSTOM_DIMS)
    spec, effective = template.build(template.defaults(), 32)
    assert effective["channels_used"] == 6
    assert any(node["type"] == "preprocess.channel_select" for node in spec["nodes"])


def test_search_space_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="不存在的节点"):
        normalize_search_space(_custom_spec(), [{"node_id": "missing", "param": "high"}])
    with pytest.raises(ValueError, match="没有参数"):
        normalize_search_space(_custom_spec(), [{"node_id": "bp", "param": "nope"}])
    with pytest.raises(ValueError, match="不能小于"):
        normalize_search_space(
            _custom_spec(),
            [{"node_id": "bp", "param": "high", "low": 0.1, "high": 40}],
        )
    with pytest.raises(ValueError, match="不在 \\["):
        normalize_search_space(
            _custom_spec(),
            [{"node_id": "csp", "param": "n_components", "low": 6, "high": 8}],
        )
    riemann = {
        "nodes": [
            {"id": "bp", "type": "preprocess.bandpass", "params": {"low": 8, "high": 30}},
            {"id": "mdm", "type": "decoder.riemann_mdm", "params": {}},
        ],
        "edges": [{"source": "bp", "target": "mdm"}],
    }
    with pytest.raises(ValueError, match="当前值"):
        normalize_search_space(
            riemann,
            [{"node_id": "mdm", "param": "metric", "options": ["logeuclid", "euclid"]}],
        )
    with pytest.raises(ValueError, match="重复"):
        normalize_search_space(
            _custom_spec(),
            [{"node_id": "bp", "param": "high", "low": 20, "high": 40},
             {"node_id": "bp", "param": "high", "low": 20, "high": 40}],
        )


def test_filter_bank_band_count_cannot_be_searched():
    spec = {
        "nodes": [
            {"id": "fb", "type": "preprocess.filter_bank_split", "params": {"n_bands": 4}},
            {"id": "power", "type": "feature.bandpower", "params": {"low": 8, "high": 12}},
            {"id": "lda", "type": "decoder.lda", "params": {}},
        ],
        "edges": [
            {"source": "fb", "target": "power", "source_port": "band_1"},
            {"source": "power", "target": "lda"},
        ],
    }
    with pytest.raises(ValueError, match="输出端口"):
        normalize_search_space(
            spec,
            [{"node_id": "fb", "param": "n_bands", "low": 2, "high": 8}],
        )


def test_history_vector_preserves_precision_for_full_reevaluation():
    dims = [{"node_id": "csp", "param": "reg", "low": 1e-6, "high": 0.1, "step": 1e-6}]
    template = custom_template(9, "精度回归", "", _custom_spec(), dims)
    vector = template.default_vector()
    params = _decode(template.space, vector)
    assert params["csp::reg"] == pytest.approx(1e-6)
    record = _candidate_record(
        0, "proxy", 0, "init", vector, params, None, None, 0.0, []
    )
    assert _decode(template.space, np.asarray(record["vector"])) == params


def test_run_hpo_with_custom_template(registry):
    template = custom_template(7, "我的丹方", "", _custom_spec(), _CUSTOM_DIMS[:2])
    result = run_hpo(registry, _request(budget=6, population=3, full_candidates=2),
                     template=template)
    assert result["template"]["id"] == "custom:7"
    assert result["template"]["source"]["pipeline_id"] == 7
    assert len(result["template"]["space"]) == 2
    assert result["best"]["graph"]
    assert {key for key in result["best"]["params"] if "::" in key} == {
        "bp::high", "csp::n_components",
    }


def _guard_spec(low, high, n_components=4, reg=1e-6, resample=None):
    nodes = []
    if resample is not None:
        nodes.append(
            {"id": "rs", "type": "preprocess.resample",
             "params": {"target_sfreq": resample}}
        )
    nodes += [
        {"id": "bp", "type": "preprocess.bandpass", "params": {"low": low, "high": high}},
        {"id": "csp", "type": "feature.csp",
         "params": {"n_components": n_components, "reg": reg}},
    ]
    return {"nodes": nodes}


def test_candidate_guardrails_reject_invalid_bands_and_csp():
    _validate_candidate(_guard_spec(8, 30), 8, 128)
    with pytest.raises(ValueError, match="下界"):
        _validate_candidate(_guard_spec(30, 8), 8, 128)
    with pytest.raises(ValueError, match="Nyquist"):
        _validate_candidate(_guard_spec(8, 80), 8, 128)
    with pytest.raises(ValueError, match="Nyquist"):
        _validate_candidate(_guard_spec(8, 30, resample=40), 8, 128)
    with pytest.raises(ValueError, match="通道数"):
        _validate_candidate(_guard_spec(8, 30, n_components=12), 8, 128)
    with pytest.raises(ValueError, match="reg"):
        _validate_candidate(_guard_spec(8, 30, reg=0), 8, 128)


def test_run_hpo_aggregates_candidate_failures(registry):
    spec = _custom_spec()
    spec["nodes"][0]["params"]["high"] = 90
    dims = [{"node_id": "bp", "param": "high", "low": 70, "high": 120, "step": 1}]
    template = custom_template(11, "越界频带", "", spec, dims)
    result = run_hpo(
        registry,
        _request(budget=6, population=3, full_candidates=1),
        template=template,
    )
    failures = result["failures"]
    assert failures["evaluations"] == 3
    assert failures["failed"] == 3
    assert result["search"]["failed_evaluations"] == 3
    assert result["search"]["evaluations"] == 3
    assert result["best"] is None
    assert len(failures["groups"]) == 1
    assert "Nyquist" in failures["groups"][0]["reason"]
    errors = [item["error"] for item in result["history"]]
    assert len(errors) == 3 and all("Nyquist" in error for error in errors)


def test_failure_summary_groups_numeric_variants():
    history = [
        {"error": "ValueError: bp 的上界 80 Hz 超过 Nyquist 频率 64 Hz"},
        {"error": "ValueError: bp 的上界 90 Hz 超过 Nyquist 频率 64 Hz"},
        {"error": "RuntimeError: 矩阵分解失败"},
        {"error": None},
    ]
    summary = _failure_summary(history)
    assert summary["evaluations"] == 4
    assert summary["failed"] == 3
    assert summary["groups"][0]["count"] == 2
    assert "Nyquist" in summary["groups"][0]["reason"]
