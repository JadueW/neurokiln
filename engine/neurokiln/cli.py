from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .bundle import export_bundle, reproduce_bundle, reproduce_experiment
from .data import DatasetRegistry
from .online import ThresholdPolicy, replay_from_split
from .plugins import PluginManager, validate_plugin_source
from .protocol import build_protocol
from .report import build_html_report, build_workbook
from .runner import ExperimentRequest, resolve_pipelines, run_experiment
from .store import ExperimentStore


def _default_data_root() -> Path:
    configured = os.environ.get("NEUROKILN_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [
        Path.cwd() / "data",
        Path(__file__).resolve().parents[2] / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _registry(args) -> DatasetRegistry:
    return DatasetRegistry(
        Path(args.data_dir).expanduser().resolve(),
        mne_data_dir=args.mne_data or os.environ.get("NEUROKILN_MNE_DATA") or None,
    )


def _store(args) -> ExperimentStore:
    path = args.db or (Path(args.data_dir) / "neurokiln.db")
    return ExperimentStore(Path(path).expanduser().resolve())


def _plugins(args) -> PluginManager:
    return PluginManager(Path(args.data_dir).expanduser().resolve() / "plugins")


def _split_list(values: list[str] | None, cast=str) -> list:
    result = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                result.append(cast(part))
    return result


def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    line = "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def _print_ba(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _print_ci(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "—"
    if value[0] is None or value[1] is None:
        return "—"
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"rows", "manifest"}
    } | {
        "rows": [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "splits"
                },
                "splits": [
                    {
                        key: value
                        for key, value in split.items()
                        if key not in {"y_true", "y_pred", "y_proba", "proba_classes"}
                    }
                    for split in row.get("splits", [])
                ],
            }
            for row in result.get("rows", [])
        ]
    }


def _progress_printer(enabled: bool):
    last = {"line": ""}

    def callback(done: int, total: int, info: dict[str, Any]) -> None:
        if not enabled:
            return
        line = (
            f"[{done}/{total}] {info.get('pipeline', '')} "
            f"fold {int(info.get('split_index', 0)) + 1}/{info.get('n_splits', '?')}"
        )
        sys.stderr.write("\r" + line.ljust(len(last["line"])))
        sys.stderr.flush()
        last["line"] = line

    return callback


def _finish_progress(enabled: bool) -> None:
    if enabled:
        sys.stderr.write("\n")
        sys.stderr.flush()


def command_cache(args) -> int:
    cache = _registry(args).artifact_cache
    stats = cache.clear() if args.clear else cache.stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    print(
        f"缓存 {'已清空' if args.clear else '状态'} · {stats['entries']} 项 / "
        f"{stats['bytes'] / 1024 / 1024:.1f} MiB · 命中率 {stats['hit_rate']:.1%}"
    )
    print(f"目录: {stats['root']}")
    print(
        f"命中 {stats['hits']} · 未命中 {stats['misses']} · 写入 {stats['writes']} · "
        f"淘汰 {stats['evictions']}"
    )
    return 0


def command_list_datasets(args) -> int:
    registry = _registry(args)
    items = registry.describe(args.dataset_id)
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    rows = [
        [
            item["id"],
            item["adapter"],
            item["paradigm"],
            item["classes"],
            item["channels"],
            item["subject_count"],
            item["status"],
        ]
        for item in items
    ]
    _print_table(["ID", "适配器", "范式", "类别", "通道", "被试", "状态"], rows)
    return 0


def _run(args, single: bool = False) -> int:
    pipelines = _split_list(args.pipeline)
    if single and len(pipelines) > 1:
        raise SystemExit("`run` 只接受一个 --pipeline；多算法对比请用 `arena`")
    registry = _registry(args)
    store = None if args.no_save else _store(args)
    seeds = _split_list(args.seeds, int) or [0]
    request = ExperimentRequest(
        dataset_id=args.dataset,
        subject_ids=_split_list(args.subjects),
        classes=args.classes,
        options={},
        protocol=args.protocol,
        folds=args.folds,
        repeats=args.repeats,
        seeds=seeds,
        pipelines=pipelines,
        name=args.name or "",
        notes=args.notes or "",
        save=not args.no_save,
        use_cache=not args.no_cache,
    )
    result = run_experiment(
        registry,
        store,
        request,
        on_progress=_progress_printer(not args.json),
        plugins=_plugins(args),
    )
    _finish_progress(not args.json)
    if args.export_dir:
        if result["experiment_id"] is None:
            raise SystemExit("导出可复现包需要保存实验；请去掉 --no-save")
        bundle = export_bundle(store, result["experiment_id"], args.export_dir)
        result["bundle"] = bundle
    if args.json:
        print(json.dumps(_public_result(result), ensure_ascii=False, indent=2))
        return 0
    stats = result["statistics"]
    print(
        f"实验 #{result['experiment_id']} · {result['name']} · "
        f"{result['n_splits']} 折 · 检验单位 {stats.get('unit')} "
        f"(n={stats.get('n_units')}) · {result['duration_ms'] / 1000:.1f}s"
    )
    print(stats.get("note", ""))
    ci_map = (stats.get("ci") or {}).get("pipelines") or {}
    rows = []
    for row in result["rows"]:
        interval = row.get("balanced_accuracy_ci95") or (ci_map.get(row["pipeline"]) or {}).get("ci95")
        rows.append(
            [
                row.get("pipeline_name") or row["pipeline"],
                _print_ba(row.get("balanced_accuracy_mean")),
                f"± {_print_ba(row.get('balanced_accuracy_std'))}",
                _print_ci(interval),
                _print_ba(row.get("cohen_kappa_mean")),
                len(row.get("splits") or []),
            ]
        )
    _print_table(["丹方", "BA", "SD", "95% CI", "Kappa", "折数"], rows)
    if args.export_dir:
        print(f"可复现包: {result['bundle']['directory']}")
    return 0


def command_run(args) -> int:
    return _run(args, single=True)


def command_arena(args) -> int:
    return _run(args, single=False)


def command_report(args) -> int:
    store = _store(args)
    detail = store.get_experiment(int(args.experiment))
    if detail is None:
        raise SystemExit(f"实验 {args.experiment} 不存在")
    html_path = Path(args.out or f"report-{args.experiment}.html").expanduser()
    html_path.write_text(build_html_report(detail), encoding="utf-8")
    outputs = {"html": str(html_path.resolve())}
    if args.xlsx:
        xlsx_path = Path(args.xlsx).expanduser()
        xlsx_path.write_bytes(build_workbook(detail))
        outputs["xlsx"] = str(xlsx_path.resolve())
    if args.bundle:
        outputs["bundle"] = export_bundle(store, int(args.experiment), args.bundle)["directory"]
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        for kind, path in outputs.items():
            print(f"{kind}: {path}")
    return 0


def command_export(args) -> int:
    store = _store(args)
    bundle = export_bundle(
        store, int(args.experiment), args.out,
        include_report=not args.no_report,
    )
    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        print(f"可复现包: {bundle['directory']}")
        print(f"预测: {bundle['predictions']}")
        for kind, path in bundle["reports"].items():
            print(f"{kind}: {path}")
    return 0


def command_reproduce(args) -> int:
    registry = _registry(args)
    store = _store(args)
    target = Path(args.experiment).expanduser()
    plugins = _plugins(args)
    if target.exists():
        result = reproduce_bundle(
            registry,
            target,
            store=store,
            save=args.save,
            tolerance=args.tolerance,
            allow_data_mismatch=args.allow_data_mismatch,
            plugins=plugins,
        )
    else:
        try:
            experiment_id = int(args.experiment)
        except ValueError as exc:
            raise SystemExit(f"无法识别的实验或目录: {args.experiment}") from exc
        result = reproduce_experiment(
            registry,
            store,
            experiment_id,
            save=args.save,
            tolerance=args.tolerance,
            allow_data_mismatch=args.allow_data_mismatch,
            plugins=plugins,
        )
    if args.json:
        print(json.dumps(
            {
                "matched": result["matched"],
                "tolerance": result["tolerance"],
                "data_signature": result["data_signature"],
                "expected_signature": result["expected_signature"],
                "comparisons": result["comparisons"],
                "experiment_id": result["experiment_id"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if result["matched"] else 1
    rows = [
        [
            item["pipeline_name"],
            _print_ba(item["expected_balanced_accuracy"]),
            _print_ba(item["actual_balanced_accuracy"]),
            _print_ba(item["difference"]),
            "通过" if item["matched"] else "不一致",
        ]
        for item in result["comparisons"]
    ]
    print(f"复现结果: {'通过' if result['matched'] else '不一致'}（容差 {result['tolerance']}）")
    _print_table(["丹方", "基准 BA", "复现 BA", "差值", "结论"], rows)
    if result["experiment_id"]:
        print(f"复现实验已入库: #{result['experiment_id']}")
    return 0 if result["matched"] else 1


def command_plugin_validate(args) -> int:
    report = validate_plugin_source(args.path)
    failed = not report["valid"] or (args.strict and bool(report["warnings"]))
    if args.json:
        print(json.dumps({**report, "strict": args.strict}, ensure_ascii=False, indent=2))
        return 0 if not failed else 1
    print(f"插件校验: {'通过' if not failed else '失败'}（{report['kind']}）")
    if report["manifest"]:
        item = report["manifest"]
        print(
            f"  {item['name']} {item['version']} | {item['runtime']} | "
            f"清单 v{item['schema_version']} | {report['compatibility']}"
        )
    package = report["package"]
    if package["declared"]:
        print(
            f"  文件哈希: {'已验证' if package['verified'] else '不一致'}"
            f"（{package['files']} 个文件）"
        )
    for pipeline in report["pipelines"]:
        print(f"  图 {pipeline['id']}: {pipeline['status']}")
    for line in report["warnings"]:
        print(f"  警告: {line}")
    for line in report["errors"]:
        print(f"  错误: {line}")
    return 0 if not failed else 1


def command_online_replay(args) -> int:
    registry = _registry(args)
    try:
        options = json.loads(args.options) if args.options else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--options 不是有效 JSON: {exc}") from exc
    epochs = registry.load(
        args.dataset,
        subject_ids=_split_list(args.subjects) or None,
        classes=args.classes,
        options=options,
    )
    splits = build_protocol(
        args.protocol,
        folds=args.folds,
        repeats=1,
        seeds=(int(args.seed),),
    ).split(epochs)
    if not 0 <= args.fold < len(splits):
        raise SystemExit(f"--fold 超出范围：可用 0..{len(splits) - 1}")
    split = splits[args.fold]
    definitions, pipelines = resolve_pipelines(
        [args.pipeline], _store(args), _plugins(args)
    )
    result = pipelines[0].run_split(epochs, split)
    closed = replay_from_split(
        epochs,
        split,
        result,
        policy=ThresholdPolicy(
            threshold=args.threshold,
            action_template=args.action_template,
        ),
        hop_seconds=args.hop_seconds,
    )
    payload = {
        "dataset": args.dataset,
        "pipeline": definitions[0]["name"],
        "protocol": args.protocol,
        "fold": args.fold,
        "calibration_trials": int(split.train.size),
        "stream_trials": int(split.test.size),
        **closed.to_payload(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"在线回放 · {definitions[0]['name']} · {args.protocol} 第 {args.fold} 折")
    print(
        f"校准 {payload['calibration_trials']} 试次 → 流式 {payload['stream_trials']} 试次"
    )
    print(
        f"动作 {payload['n_actions']} · 弃权 {payload['n_abstained']} · "
        f"逐试次准确率 {_print_ba(payload['accuracy'])}"
    )
    for event in payload["events"][:10]:
        truth = "-" if event["true_label"] is None else event["true_label"]
        print(
            f"  {event['trial_id']}  t={event['timestamp']:.2f}s  "
            f"true={truth} pred={event['predicted_label']}  "
            f"conf={event['confidence'] if event['confidence'] is None else round(event['confidence'], 3)}  "
            f"action={event['action']}"
        )
    if len(payload["events"]) > 10:
        print(f"  ... 其余 {len(payload['events']) - 10} 个事件省略")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurokiln",
        description="NeuroKiln 命令行：真实数据、固定协议、可复现评测。",
    )
    parser.add_argument("--data-dir", default=str(_default_data_root()), help="数据根目录")
    parser.add_argument("--db", default="", help="实验库路径（默认 <data-dir>/neurokiln.db）")
    parser.add_argument("--mne-data", default="", help="MOABB/MNE 数据缓存目录")
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("list-datasets", help="列出可用数据集")
    datasets.add_argument("--dataset", dest="dataset_id", default=None)
    datasets.add_argument("--json", action="store_true")
    datasets.set_defaults(func=command_list_datasets)

    cache = subparsers.add_parser("cache", help="查看或清空预处理产物缓存")
    cache.add_argument("--clear", action="store_true")
    cache.add_argument("--json", action="store_true")
    cache.set_defaults(func=command_cache)

    for name, function, help_text in (
        ("run", command_run, "运行单条丹方"),
        ("arena", command_arena, "运行多条丹方对比"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--dataset", default="synthetic")
        sub.add_argument("--subjects", action="append", default=[], help="被试，逗号分隔，如 S01,S02")
        sub.add_argument("--classes", type=int, default=None)
        sub.add_argument("--protocol", default="within_subject",
                         choices=["within_subject", "cross_session", "loso"])
        sub.add_argument("--folds", type=int, default=5)
        sub.add_argument("--repeats", type=int, default=1)
        sub.add_argument("--seeds", action="append", default=[], help="随机种子，逗号分隔，默认 0")
        sub.add_argument("--pipeline", action="append", required=True,
                         help="丹方 ID（可重复），自定义用 custom:<id>")
        sub.add_argument("--name", default="")
        sub.add_argument("--notes", default="")
        sub.add_argument("--no-save", action="store_true", help="不写入实验库")
        sub.add_argument("--no-cache", action="store_true", help="禁用预处理产物缓存")
        sub.add_argument("--export-dir", default="", help="保存后同时导出可复现包")
        sub.add_argument("--json", action="store_true")
        if name == "arena":
            sub.set_defaults(func=function)
        else:
            sub.set_defaults(func=function)

    report = subparsers.add_parser("report", help="生成 HTML / XLSX 报告")
    report.add_argument("--experiment", type=int, required=True)
    report.add_argument("--out", default="")
    report.add_argument("--xlsx", default="")
    report.add_argument("--bundle", default="")
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=command_report)

    export = subparsers.add_parser("export", help="导出可复现包")
    export.add_argument("--experiment", type=int, required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--no-report", action="store_true")
    export.add_argument("--json", action="store_true")
    export.set_defaults(func=command_export)

    online = subparsers.add_parser(
        "online-replay", help="用留出折回放逐试次预测与闭环动作（在线扩展点）"
    )
    online.add_argument("--dataset", default="synthetic")
    online.add_argument("--subjects", action="append", default=[], help="被试，逗号分隔")
    online.add_argument("--classes", type=int, default=None)
    online.add_argument("--options", default="", help="数据集选项 JSON")
    online.add_argument("--pipeline", default="csp_lda", help="丹方 ID")
    online.add_argument("--protocol", default="within_subject",
                        choices=["within_subject", "cross_session", "loso"])
    online.add_argument("--folds", type=int, default=5)
    online.add_argument("--fold", type=int, default=0)
    online.add_argument("--seed", type=int, default=0)
    online.add_argument("--threshold", type=float, default=0.6)
    online.add_argument("--action-template", default="select:{label}")
    online.add_argument("--hop-seconds", type=float, default=1.0)
    online.add_argument("--json", action="store_true")
    online.set_defaults(func=command_online_replay)

    reproduce = subparsers.add_parser("reproduce", help="按实验 ID 或复现包重跑并比对")
    reproduce.add_argument("experiment", help="实验 ID 或 bundle 目录")
    reproduce.add_argument("--tolerance", type=float, default=0.02)
    reproduce.add_argument("--save", action="store_true")
    reproduce.add_argument("--allow-data-mismatch", action="store_true")
    reproduce.add_argument("--json", action="store_true")
    reproduce.set_defaults(func=command_reproduce)

    plugin = subparsers.add_parser("plugin", help="插件一致性校验")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    validate = plugin_sub.add_parser(
        "validate", help="校验插件目录 / ZIP / plugin.json，不安装、不执行代码"
    )
    validate.add_argument("path", help="插件目录、ZIP 包或 plugin.json")
    validate.add_argument("--strict", action="store_true", help="警告也视为失败")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=command_plugin_validate)
    return parser


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
