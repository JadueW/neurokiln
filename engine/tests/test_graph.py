import numpy as np
import pytest

from neurokiln.cache import ArtifactCache
from neurokiln.evaluate import balanced_accuracy, run_arena
from neurokiln.graph import (
    GraphPipeline,
    GraphValidationError,
    build_graph_pipeline,
    node_catalog,
)
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol
from test_baselines import make_oscillatory_epochs


def _node(node_id, node_type, x=0, y=0, **params):
    return {"id": node_id, "type": node_type, "params": params, "x": x, "y": y}


def _linear_csp_spec():
    return {
        "nodes": [
            _node("n1", "preprocess.bandpass", 60, 120, low=8, high=30, order=4),
            _node("n2", "feature.csp", 250, 120, n_components=6, reg=1e-6),
            _node("n3", "feature.standardize", 440, 120),
            _node("n4", "decoder.lda", 620, 120, shrinkage="auto"),
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
        ],
    }


def test_node_catalog_is_serializable():
    catalog = node_catalog()
    assert {item["type"] for item in catalog} >= {
        "preprocess.bandpass", "feature.csp", "merge.concat", "decoder.eegnet",
    }
    for item in catalog:
        assert item["label"] and item["kind"] in {"preprocess", "feature", "merge", "decoder"}
        for param in item["params"]:
            assert {"name", "label", "type", "default"}.issubset(param)


def test_linear_graph_matches_builtin_pipeline():
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    split = WithinSubjectProtocol(folds=4, seeds=(2,)).split(epochs)[0]
    graph = build_graph_pipeline("custom:1", _linear_csp_spec())
    builtin = build_baseline("csp_lda")
    graph_result = graph.run_split(epochs, split)
    builtin_result = builtin.run_split(epochs, split)
    np.testing.assert_array_equal(graph_result["y_pred"], builtin_result["y_pred"])
    assert balanced_accuracy(graph_result["y_true"], graph_result["y_pred"]) > 0.8


def test_branch_merge_graph_runs_through_arena():
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    spec = {
        "nodes": [
            _node("n1", "preprocess.bandpass", 40, 100, low=8, high=30),
            _node("n2", "feature.csp", 220, 40, n_components=4),
            _node("n3", "feature.channel_std", 220, 190),
            _node("n4", "merge.concat", 410, 110),
            _node("n5", "feature.standardize", 560, 110),
            _node("n6", "decoder.lda", 720, 110),
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n1", "target": "n3"},
            {"source": "n2", "target": "n4"},
            {"source": "n3", "target": "n4"},
            {"source": "n4", "target": "n5"},
            {"source": "n5", "target": "n6"},
        ],
    }
    graph = GraphPipeline("custom:merge", spec)
    assert graph.spec["decoder"] == "n6"
    assert graph.order == ["n1", "n2", "n3", "n4", "n5", "n6"]
    splits = WithinSubjectProtocol(folds=4, seeds=(1,)).split(epochs)
    rows = run_arena(epochs, [graph], splits)
    assert rows[0]["pipeline"] == "custom:merge"
    assert rows[0]["balanced_accuracy_mean"] > 0.8


def test_graph_rejects_cycles_multiple_decoders_and_bad_merges():
    with pytest.raises(GraphValidationError, match="环"):
        build_graph_pipeline(
            "cycle",
            {
                "nodes": [
                    _node("n1", "preprocess.bandpass"),
                    _node("n2", "feature.csp"),
                    _node("n3", "decoder.lda"),
                ],
                "edges": [
                    {"source": "n1", "target": "n2"},
                    {"source": "n2", "target": "n1"},
                    {"source": "n2", "target": "n3"},
                ],
            },
        )
    with pytest.raises(GraphValidationError, match="一个分类器"):
        build_graph_pipeline(
            "two-decoders",
            {
                "nodes": [
                    _node("n1", "feature.channel_mean"),
                    _node("n2", "decoder.lda"),
                    _node("n3", "decoder.nearest_centroid"),
                ],
                "edges": [
                    {"source": "n1", "target": "n2"},
                    {"source": "n1", "target": "n3"},
                ],
            },
        )
    with pytest.raises(GraphValidationError, match="两个输入"):
        build_graph_pipeline(
            "merge-one-input",
            {
                "nodes": [
                    _node("n1", "feature.channel_mean"),
                    _node("n2", "merge.concat"),
                    _node("n3", "decoder.lda"),
                ],
                "edges": [
                    {"source": "n1", "target": "n2"},
                    {"source": "n2", "target": "n3"},
                ],
            },
        )


def _fusion_spec(fusion_type="merge.average"):
    return {
        "nodes": [
            _node("n1", "preprocess.bandpass", 40, 100, low=8, high=30),
            _node("n2", "feature.csp", 220, 40, n_components=4),
            _node("n3", "feature.standardize", 380, 40),
            _node("n4", "decoder.lda", 540, 40),
            _node("n5", "preprocess.bandpass", 220, 200, low=16, high=30),
            _node("n6", "feature.channel_std", 380, 200),
            _node("n7", "decoder.nearest_centroid", 540, 200),
            _node("n8", fusion_type, 720, 120),
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
            {"source": "n1", "target": "n5"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
            {"source": "n4", "target": "n8"},
            {"source": "n7", "target": "n8"},
        ],
    }


def test_average_fusion_combines_decoder_probabilities():
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    split = WithinSubjectProtocol(folds=4, seeds=(1,)).split(epochs)[0]
    graph = build_graph_pipeline("custom:fusion", _fusion_spec())
    assert graph.spec["decoders"] == ["n4", "n7"]
    assert graph.spec["prediction"] == "n8"
    result = graph.run_split(epochs, split)
    assert result["y_pred"].shape == epochs.select(split.test).labels.shape
    np.testing.assert_allclose(result["y_proba"].sum(axis=1), 1.0, atol=1e-9)
    assert balanced_accuracy(result["y_true"], result["y_pred"]) > 0.8
    rows = run_arena(epochs, [graph], [split])
    assert rows[0]["balanced_accuracy_mean"] > 0.8


def test_vote_fusion_is_terminal_and_requires_all_decoders():
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    split = WithinSubjectProtocol(folds=4, seeds=(2,)).split(epochs)[0]
    graph = build_graph_pipeline("custom:vote", _fusion_spec("merge.vote"))
    result = graph.run_split(epochs, split)
    assert balanced_accuracy(result["y_true"], result["y_pred"]) > 0.8
    spec = _fusion_spec()
    spec["nodes"].append(_node("n9", "feature.standardize"))
    spec["edges"].append({"source": "n8", "target": "n9"})
    with pytest.raises(GraphValidationError, match="融合节点必须是最终输出"):
        build_graph_pipeline("fusion-tail", spec)
    spec = _fusion_spec()
    spec["nodes"].pop()  # remove fusion, leaving two independent decoders
    spec["edges"] = [edge for edge in spec["edges"] if edge["target"] != "n8"]
    with pytest.raises(GraphValidationError, match="一个分类器"):
        build_graph_pipeline("no-fusion", spec)


def _filter_bank_spec():
    return {
        "nodes": [
            _node("n1", "preprocess.filter_bank_split", 40, 100, low=8, high=32, n_bands=3),
            _node("n2", "preprocess.car", 200, 40),
            _node("n3", "preprocess.car", 200, 180),
            _node("n4", "feature.bandpower", 360, 40, low=8, high=16),
            _node("n5", "feature.bandpower", 360, 180, low=16, high=24),
            _node("n6", "merge.concat", 520, 110),
            _node("n7", "decoder.lda", 680, 110),
        ],
        "edges": [
            {"source": "n1", "target": "n2", "source_port": "band_1"},
            {"source": "n1", "target": "n3", "source_port": "band_2"},
            {"source": "n2", "target": "n4"},
            {"source": "n3", "target": "n5"},
            {"source": "n4", "target": "n6"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
        ],
    }


def test_filter_bank_split_exposes_named_ports_and_isolates_cache(tmp_path):
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    split = WithinSubjectProtocol(folds=4, seeds=(3,)).split(epochs)[0]
    graph = build_graph_pipeline("custom:ports", _filter_bank_spec())
    ports = graph.spec["edges"][:2]
    assert [edge["source_port"] for edge in ports] == ["band_1", "band_2"]
    cache = ArtifactCache(tmp_path / "cache")
    first = graph.run_split(epochs, split, cache=cache)
    second = graph.run_split(epochs, split, cache=cache)
    np.testing.assert_array_equal(first["y_pred"], second["y_pred"])
    assert balanced_accuracy(first["y_true"], first["y_pred"]) > 0.8
    assert cache.stats()["entries"] == 4

    missing_port = _filter_bank_spec()
    missing_port["edges"][0].pop("source_port")
    with pytest.raises(GraphValidationError, match="source_port"):
        build_graph_pipeline("ports-missing", missing_port)
    bad_port = _filter_bank_spec()
    bad_port["edges"][0]["source_port"] = "band_9"
    with pytest.raises(GraphValidationError, match="没有输出端口"):
        build_graph_pipeline("ports-bad", bad_port)


def test_graph_coerces_and_validates_parameters():
    graph = build_graph_pipeline(
        "coerced",
        {
            "nodes": [
                _node("n1", "feature.csp", n_components="6"),
                _node("n2", "decoder.lda", shrinkage="0.2"),
            ],
            "edges": [{"source": "n1", "target": "n2"}],
        },
    )
    assert graph.nodes["n1"].n_components == 6
    with pytest.raises(GraphValidationError, match="未知参数"):
        build_graph_pipeline(
            "bad-param",
            {
                "nodes": [_node("n1", "feature.csp", bogus=1), _node("n2", "decoder.lda")],
                "edges": [{"source": "n1", "target": "n2"}],
            },
        )


def test_catalog_declares_node_input_modalities():
    catalog = {item["type"]: item for item in node_catalog()}
    assert catalog["decoder.eegnet"]["inputs"] == ["epochs"]
    assert catalog["decoder.riemann_mdm"]["inputs"] == ["epochs"]
    assert catalog["decoder.lda"]["inputs"] == ["features"]
    assert catalog["feature.select_k"]["inputs"] == ["features"]
    assert catalog["feature.csp"]["inputs"] == ["epochs"]


def _chain_spec(nodes, pairs):
    return {
        "nodes": nodes,
        "edges": [{"source": source, "target": target} for source, target in pairs],
    }


def test_feature_output_cannot_feed_epoch_only_decoder():
    spec = _chain_spec(
        [
            _node("bp", "preprocess.bandpass", low=8, high=30),
            _node("power", "feature.bandpower", low=8, high=30),
            _node("select", "feature.select_k", k=4),
            _node("deep", "decoder.eegnet"),
        ],
        [("bp", "power"), ("power", "select"), ("select", "deep")],
    )
    with pytest.raises(GraphValidationError, match="只接受 3D 试次") as excinfo:
        build_graph_pipeline("feature-to-eegnet", spec)
    assert "EEGNet" in str(excinfo.value)
    assert "LDA 或最近质心" in str(excinfo.value)


def test_features_cannot_feed_epoch_only_feature_node():
    spec = _chain_spec(
        [
            _node("bp", "preprocess.bandpass", low=8, high=30),
            _node("mean", "feature.channel_mean"),
            _node("csp", "feature.csp", n_components=4),
            _node("lda", "decoder.lda"),
        ],
        [("bp", "mean"), ("mean", "csp"), ("csp", "lda")],
    )
    with pytest.raises(GraphValidationError, match="只接受 3D 试次"):
        build_graph_pipeline("features-to-csp", spec)
