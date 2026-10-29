from io import BytesIO

import pytest

from neurokiln.report import _tex, build_docx, build_latex


@pytest.fixture()
def detail():
    return {
        "id": 42,
        "name": "CSP vs Riemann 对照",
        "status": "completed",
        "protocol": "within_subject",
        "protocol_params": {"folds": 2, "repeats": 1, "seeds": [0]},
        "notes": "回归样例",
        "engine_version": "0.4.0",
        "duration_ms": 1234.0,
        "seeds": [0],
        "n_runs": 2,
        "dataset": {
            "id": "synthetic",
            "name": "合成数据",
            "version": "1",
            "signature": "sig_abc-123",
            "manifest": {
                "subjects": ["S01", "S02"],
                "loaded_n_trials": 40,
                "loaded_channels": 8,
                "loaded_sfreq": 128,
                "label_names": ["左手", "右手"],
                "citation": "NeuroKiln synthetic generator",
            },
        },
        "rows": [
            {
                "pipeline": "csp_lda",
                "pipeline_name": "CSP + LDA",
                "balanced_accuracy_mean": 0.81,
                "balanced_accuracy_std": 0.02,
                "balanced_accuracy_ci95": [0.78, 0.84],
                "cohen_kappa_mean": 0.62,
                "calibration": {"ece": 0.05},
                "splits": [
                    {
                        "duration_ms": 12.5,
                        "y_true": [0, 1, 0, 1],
                        "y_pred": [0, 1, 0, 0],
                    }
                ],
            },
            {
                "pipeline": "riemann_mdm",
                "pipeline_name": "Riemann MDM",
                "balanced_accuracy_mean": 0.74,
                "balanced_accuracy_std": 0.03,
                "cohen_kappa_mean": 0.48,
                "calibration": {"ece": 0.08},
                "splits": [{"duration_ms": 9.0, "y_true": [0, 1], "y_pred": [0, 1]}],
            },
        ],
        "statistics": {
            "unit": "subject",
            "n_units": 2,
            "n_splits": 2,
            "alpha": 0.05,
            "friedman": {"available": True, "statistic": 3.0, "p_value": 0.083,
                         "significant": False},
            "nemenyi": {"available": True, "critical_difference": 1.96},
            "pairwise": [
                {"pipeline_a": "csp_lda", "pipeline_b": "riemann_mdm", "statistic": 1.0,
                 "p_value": 0.5, "p_value_adjusted": 0.5, "significant": False}
            ],
        },
    }


def test_latex_report_escapes_and_lists_gbt_references(detail):
    text = build_latex(detail)
    assert text.startswith("% 使用 XeLaTeX 编译")
    assert r"\begin{document}" in text and r"\end{document}" in text
    assert r"\usepackage{booktabs}" in text
    assert r"95\% CI" in text
    assert r"sig\_abc-123" in text
    assert "CSP + LDA" in text
    assert r"JAYARAM V, BARACHANT A" in text
    assert r"\bibitem" in text
    assert "NeuroKiln synthetic generator" in text
    assert "平衡准确率" in text


def test_docx_report_contains_title_tables_and_references(detail):
    payload = build_docx(detail)
    assert payload[:2] == b"PK"
    from docx import Document

    document = Document(BytesIO(payload))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "NeuroKiln 实验报告" in text
    assert "参考文献" in text
    assert "JAYARAM V, BARACHANT A" in text
    assert len(document.tables) >= 3
    table_text = "\n".join(
        cell.text for table in document.tables for row in table.rows for cell in row.cells
    )
    assert "CSP + LDA" in table_text
    assert "0.8100" in table_text


def test_tex_escaping_handles_special_characters():
    assert _tex("a_b & 50% #1 {x} ~ ^ \\") == (
        r"a\_b \& 50\% \#1 \{x\} \textasciitilde{} \textasciicircum{} \textbackslash{}"
    )
