"""Regenerate the read-only demo under web/public/demo/.

The demo runs entirely on the built-in synthetic dataset with a temporary
SQLite store, so it never touches user experiments or requires downloaded data.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from neurokiln.data import DatasetRegistry  # noqa: E402
from neurokiln.report import build_html_report  # noqa: E402
from neurokiln.runner import ExperimentRequest, run_experiment  # noqa: E402
from neurokiln.store import ExperimentStore  # noqa: E402

DEMO_DIR = ROOT / "web" / "public" / "demo"

BANNER = """
<div class="demo-banner">
  <strong>只读演示</strong>
  <span>该报告使用内置合成 EEG 生成，仅用于展示评测流程，不代表真实 BCI 性能。</span>
  <a href="/demo/index.html">返回演示首页</a>
</div>
<style>
.demo-banner { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center;
  margin:0 0 16px; padding:14px 18px; border-radius:10px; background:#fff7e6;
  border:1px solid #f5dab1; color:#7a4d10; font-size:13px; }
.demo-banner strong { color:#a35c00; }
.demo-banner a { margin-left:auto; color:#2f6fb1; text-decoration:none; }
</style>
"""


def run_demo() -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as temporary:
        registry = DatasetRegistry(Path(temporary) / "data")
        store = ExperimentStore(Path(temporary) / "demo.db")
        result = run_experiment(
            registry,
            store,
            ExperimentRequest(
                dataset_id="synthetic",
                protocol="within_subject",
                folds=4,
                repeats=1,
                seeds=[0],
                pipelines=["csp_lda", "riemann_mdm"],
                options={
                    "n_trials_per_class": 20,
                    "n_times": 48,
                    "class_sep": 0.02,
                    "class_noise_step": 0.2,
                },
                name="黄金演示 · 只读报告",
                notes="内置合成 EEG，仅用于展示评测流程。",
                save=True,
                use_cache=False,
            ),
        )
        detail = store.get_experiment(result["experiment_id"])
        if detail is None:
            raise RuntimeError("demo experiment was not persisted")
        return detail, build_html_report(detail)


def landing_page(detail: dict) -> str:
    rows = sorted(
        detail.get("rows") or [],
        key=lambda row: float(row.get("balanced_accuracy_mean") or 0.0),
        reverse=True,
    )
    best = rows[0] if rows else {}
    table = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{row.get('pipeline_name') or row.get('pipeline')}</td>"
        f"<td class=\"num\">{float(row.get('balanced_accuracy_mean') or 0.0):.3f}</td>"
        f"<td class=\"num\">{float(row.get('balanced_accuracy_std') or 0.0):.3f}</td>"
        f"<td class=\"num\">{float(row.get('cohen_kappa_mean') or 0.0):.3f}</td>"
        f"<td class=\"num\">{row.get('n_splits', 0)}</td>"
        "</tr>"
        for index, row in enumerate(rows, start=1)
    )
    dataset = detail.get("dataset") or {}
    manifest = dataset.get("manifest") or {}
    summary = {
        "experiment_id": detail.get("id"),
        "dataset": dataset.get("name"),
        "protocol": detail.get("protocol"),
        "n_splits": max((row.get("n_splits", 0) for row in rows), default=0),
        "best": {
            "pipeline": best.get("pipeline"),
            "name": best.get("pipeline_name"),
            "balanced_accuracy": best.get("balanced_accuracy_mean"),
            "cohen_kappa": best.get("cohen_kappa_mean"),
        },
        "rows": [
            {
                "pipeline": row.get("pipeline"),
                "name": row.get("pipeline_name"),
                "balanced_accuracy": row.get("balanced_accuracy_mean"),
            }
            for row in rows
        ],
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroKiln 只读演示 · 脑炉</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100vh; padding: 40px 20px 60px;
  font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
  color: #eaf3ff; background: radial-gradient(1200px 600px at 80% -10%, #1c4f7d 0%, transparent 60%),
  linear-gradient(160deg, #08111f 0%, #0d2138 55%, #102c47 100%); }}
main {{ max-width: 960px; margin: 0 auto; }}
.eyebrow {{ letter-spacing: .2em; font-size: 12px; color: #7fb8ee; text-transform: uppercase; }}
h1 {{ margin: 12px 0 8px; font-size: 40px; line-height: 1.2; }}
h1 span {{ color: #8fd0ff; }}
.lede {{ margin: 0 0 26px; color: #a9c4dd; line-height: 1.8; max-width: 720px; }}
.notice {{ display: inline-block; margin-bottom: 26px; padding: 8px 14px; border-radius: 999px;
  background: rgba(245, 196, 96, .12); border: 1px solid rgba(245, 196, 96, .35); color: #f0c779;
  font-size: 13px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-bottom: 26px; }}
.metric {{ padding: 18px; border-radius: 14px; background: rgba(12, 30, 50, .66);
  border: 1px solid rgba(126, 186, 255, .18); }}
.metric .label {{ font-size: 12px; color: #89a9c7; }}
.metric .value {{ margin-top: 6px; font-size: 30px; font-weight: 700; }}
.card {{ padding: 22px; border-radius: 16px; background: rgba(9, 23, 39, .72);
  border: 1px solid rgba(126, 186, 255, .16); margin-bottom: 24px; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 11px 12px; border-bottom: 1px solid rgba(126, 186, 255, .12); text-align: left; }}
th {{ color: #8fb4d6; font-weight: 600; font-size: 12px; letter-spacing: .04em; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.button {{ display: inline-flex; align-items: center; padding: 12px 22px; border-radius: 999px;
  text-decoration: none; font-weight: 600; font-size: 14px; }}
.button.primary {{ color: #04111f; background: linear-gradient(135deg, #8fd0ff, #4f9ce6); }}
.button.ghost {{ color: #cfe4f8; border: 1px solid rgba(143, 208, 255, .35); }}
footer {{ margin-top: 34px; color: #7893ad; font-size: 12px; line-height: 1.8; }}
footer a {{ color: #8fbfe8; }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">NeuroKiln · Read-only demo</div>
  <h1>脑炉 <span>NeuroKiln</span> · 黄金演示</h1>
  <p class="lede">
    这是一份预生成的只读报告：内置合成 EEG、被试内 4 折协议，CSP+LDA 与 Riemann MDM 公平同表比较。
    它展示了数据集签名、防泄漏拟合、统计检验与多种导出格式；不需要登录或下载数据即可查看。
  </p>
  <span class="notice">合成数据仅用于演示流程，不代表真实 BCI 性能</span>
  <div class="metrics">
    <div class="metric"><div class="label">最佳 Balanced Accuracy</div>
      <div class="value">{float(best.get("balanced_accuracy_mean") or 0.0):.3f}</div></div>
    <div class="metric"><div class="label">最佳丹方</div>
      <div class="value" style="font-size:20px">{best.get("pipeline_name") or "—"}</div></div>
    <div class="metric"><div class="label">试次 / 通道</div>
      <div class="value" style="font-size:20px">{manifest.get("loaded_n_trials") or "—"} / {manifest.get("loaded_channels") or "—"}</div></div>
    <div class="metric"><div class="label">协议</div>
      <div class="value" style="font-size:20px">{detail.get("protocol")} · {manifest.get("loaded_sfreq") or "—"} Hz</div></div>
  </div>
  <div class="card">
    <table>
      <thead><tr><th>#</th><th>丹方</th><th class="num">BA</th>
      <th class="num">标准差</th><th class="num">Cohen κ</th><th class="num">折数</th></tr></thead>
      <tbody>{table}</tbody>
    </table>
  </div>
  <div class="actions">
    <a class="button primary" href="/demo/report.html">打开完整报告</a>
    <a class="button ghost" href="/">进入 NeuroKiln</a>
  </div>
  <footer>
    NeuroKiln 脑炉 · 引擎 {detail.get("engine_version") or "0.4.0"} ·
    数据签名 {dataset.get("signature") or "—"}<br>
    只读演示由 <code>scripts/export_demo.py</code> 生成，可随时重建。
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    detail, report = run_demo()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    report = report.replace("<body>", "<body>\n" + BANNER, 1)
    (DEMO_DIR / "report.html").write_text(report, encoding="utf-8")
    (DEMO_DIR / "index.html").write_text(landing_page(detail), encoding="utf-8")
    summary = {
        "experiment_id": detail.get("id"),
        "name": detail.get("name"),
        "protocol": detail.get("protocol"),
        "n_splits": max(
            (row.get("n_splits", 0) for row in detail.get("rows") or []), default=0
        ),
        "rows": [
            {
                "pipeline": row.get("pipeline"),
                "balanced_accuracy": row.get("balanced_accuracy_mean"),
            }
            for row in detail.get("rows") or []
        ],
    }
    (DEMO_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {DEMO_DIR / 'index.html'}")
    print(f"wrote {DEMO_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
