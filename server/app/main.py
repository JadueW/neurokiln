from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))

from neurokiln import __version__ as engine_version  # noqa: E402
from neurokiln.data import (  # noqa: E402
    DatasetRegistry,
    build_csv_epochs,
    build_raw_epochs,
    inspect_import_file,
    write_epochs_npz,
)
from neurokiln.evaluate import RunCancelled  # noqa: E402
from neurokiln.graph import (  # noqa: E402
    GraphValidationError,
    build_graph_pipeline,
    node_catalog,
    normalize_graph,
)
from neurokiln.hpo import (  # noqa: E402
    HPORequest,
    custom_template,
    normalize_search_space,
    objective_catalog,
    run_hpo,
    template_catalog,
)
from neurokiln.pipeline import PIPELINE_DEFINITIONS, torch_available  # noqa: E402
from neurokiln.plugins import PluginError, PluginManager  # noqa: E402
from neurokiln.protocol import build_protocol  # noqa: E402
from neurokiln.report import build_docx, build_html_report, build_latex, build_workbook  # noqa: E402
from neurokiln.runner import (  # noqa: E402
    ExperimentRequest,
    infer_spec_paradigm,
    run_experiment,
)
from neurokiln.store import ExperimentStore  # noqa: E402
from neurokiln.templates import (  # noqa: E402
    FORMAT as TEMPLATE_FORMAT,
    FORMAT_VERSION as TEMPLATE_FORMAT_VERSION,
    get_template as get_pipeline_template,
    reproduction_catalog,
    template_catalog as pipeline_template_catalog,
)

from .auth import AuthService  # noqa: E402
from .jobs import JobCancelled, JobManager  # noqa: E402

DATA_ROOT = Path(
    os.environ.get("NEUROKILN_DATA_DIR", str(ROOT / "data"))
).expanduser().resolve()
WEB_DIST = Path(
    os.environ.get("NEUROKILN_WEB_DIST", str(ROOT / "web" / "dist"))
).expanduser().resolve()
DB_PATH = Path(os.environ.get("NEUROKILN_DB", str(DATA_ROOT / "neurokiln.db")))
SECRET_PATH = Path(
    os.environ.get("NEUROKILN_AUTH_SECRET_FILE", str(DATA_ROOT / ".auth_secret"))
)
registry = DatasetRegistry(
    DATA_ROOT,
    cache_entries=int(os.environ.get("NEUROKILN_CACHE_ENTRIES", "2")),
    mne_data_dir=os.environ.get("NEUROKILN_MNE_DATA") or None,
)
store = ExperimentStore(DB_PATH)
auth = AuthService(
    DB_PATH,
    SECRET_PATH,
    ttl_seconds=int(os.environ.get("NEUROKILN_TOKEN_TTL", str(12 * 3600))),
)
jobs = JobManager(
    DB_PATH,
    max_workers=int(os.environ.get("NEUROKILN_JOB_WORKERS", "1")),
)
plugins = PluginManager(DATA_ROOT / "plugins")

app = FastAPI(title="NeuroKiln API", version="0.4.0")

PUBLIC_API_PATHS = {
    "/api/health",
    "/api/auth/status",
    "/api/auth/bootstrap",
    "/api/auth/login",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if (
        not path.startswith("/api/")
        or request.method == "OPTIONS"
        or path in PUBLIC_API_PATHS
    ):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    payload = auth.verify_token(token)
    if payload is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "未登录或登录已过期"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.user = payload
    return await call_next(request)


CORS_ORIGINS = [
    item.strip()
    for item in os.environ.get(
        "NEUROKILN_CORS_ORIGINS",
        "http://127.0.0.1:5174,http://localhost:5174,"
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

CODEX = [
    {"name": "CSP", "type": "空间滤波", "paradigm": "MI", "gpu": False,
     "params": "n_components=4~8；reg=1e-6", "cost": "低",
     "pitfall": "小样本高通道易过拟合；只能在训练折估计协方差", "ref": "Ramoser 2000"},
    {"name": "FBCSP", "type": "滤波器组特征", "paradigm": "MI", "gpu": False,
     "params": "4~8 个频段，频段宽度 4 Hz", "cost": "中",
     "pitfall": "特征选择必须嵌套在训练折内", "ref": "Ang 2008"},
    {"name": "Riemannian MDM", "type": "黎曼分类器", "paradigm": "MI", "gpu": False,
     "params": "metric=riemann；协方差估计=oas", "cost": "低",
     "pitfall": "跨被试需对齐，否则分布差异主导", "ref": "Barachant 2012"},
    {"name": "Tangent Space + LR", "type": "黎曼特征", "paradigm": "MI", "gpu": False,
     "params": "C=0.1~10；参考矩阵取训练折几何均值", "cost": "低",
     "pitfall": "强基线，深度模型应先超过它再谈创新", "ref": "Barachant 2013"},
    {"name": "EEGNet", "type": "深度解码器", "paradigm": "MI/P300/SSVEP", "gpu": True,
     "params": "F1=8,D=2,F2=16；dropout=0.5；early stopping", "cost": "高",
     "pitfall": "验证折早停；禁止用测试折选 epoch", "ref": "Lawhern 2018"},
    {"name": "ShallowConvNet", "type": "深度解码器", "paradigm": "MI", "gpu": True,
     "params": "40 filters；时域卷积 + 平方 + 对数；dropout=0.5", "cost": "中高",
     "pitfall": "小样本下需早停与正则；测试折不参与任何拟合", "ref": "Schirrmeister 2017"},
    {"name": "xDAWN + LDA", "type": "ERP 特征", "paradigm": "P300", "gpu": False,
     "params": "n_filters=4~8", "cost": "低",
     "pitfall": "按试次划分，避免同试次子样本跨折", "ref": "Rivet 2009"},
    {"name": "FBCCA / TRCA", "type": "频率识别", "paradigm": "SSVEP", "gpu": False,
     "params": "5 个子带；窗长 0.5~4 s；报告 ITR", "cost": "低",
     "pitfall": "ITR 必须写明窗长与目标数", "ref": "Chen 2015 / Nakanishi 2018"},
]

PROTOCOLS = [
    {"id": "within_subject", "name": "被试内分层 k 折", "params": "folds/repeats/seeds",
     "rule": "trial 级分层；预处理仅在训练折 fit"},
    {"id": "cross_session", "name": "跨 session", "params": "自动枚举有序 session 对",
     "rule": "train=T_i, test=T_j；禁止同一时间信息混入"},
    {"id": "loso", "name": "跨被试留一", "params": "按被试自动留一",
     "rule": "测试被试的任何 trial 不参与 fit 或对齐"},
]


class Selection(BaseModel):
    subject_ids: list[str] = Field(default_factory=list)
    classes: int | None = Field(None, ge=2, le=10)
    options: dict[str, Any] = Field(default_factory=dict)


class DatasetRequest(Selection):
    dataset_id: str = "synthetic"


class ProtocolRequest(DatasetRequest):
    protocol: str = "within_subject"
    folds: int = Field(5, ge=2, le=10)
    repeats: int = Field(1, ge=1, le=10)
    seeds: list[int] = Field(default_factory=lambda: [0])


class ArenaRequest(ProtocolRequest):
    pipelines: list[str] = Field(
        default_factory=lambda: ["csp_lda", "raw_centroid", "var_centroid"]
    )
    name: str = ""
    notes: str = ""
    save: bool = True
    use_cache: bool = True


class HPOJobRequest(ProtocolRequest):
    template_id: str = "csp_lda"
    objectives: list[str] = Field(
        default_factory=lambda: ["balanced_accuracy", "cohen_kappa"]
    )
    budget: int = Field(16, ge=4, le=120)
    population: int = Field(6, ge=3, le=16)
    proxy_folds: int = Field(2, ge=1, le=10)
    full_candidates: int = Field(3, ge=1, le=10)
    seed: int = 2026
    name: str = Field("", max_length=120)
    notes: str = Field("", max_length=1000)


class HPOPromoteRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=64)
    name: str = Field("", max_length=80)
    description: str = Field("", max_length=500)


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=200)


class UserCreate(AuthCredentials):
    role: str = "member"


class CustomPipelineRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=500)
    spec: dict[str, Any]


class SearchSpaceRequest(BaseModel):
    dims: list[dict[str, Any]] = Field(default_factory=list)


class TemplateInstallRequest(BaseModel):
    name: str = Field("", max_length=80)


class PipelineTemplateImport(BaseModel):
    format: str = TEMPLATE_FORMAT
    version: int = TEMPLATE_FORMAT_VERSION
    name: str = Field("", max_length=80)
    description: str = Field("", max_length=1000)
    spec: dict[str, Any]
    search_space: list[dict[str, Any]] = Field(default_factory=list)
    checksum: str = ""
    template: dict[str, Any] = Field(default_factory=dict)


class PluginInstallRequest(BaseModel):
    id: str = Field(min_length=2, max_length=41)
    enable: bool = True


class DatasetImportCommit(BaseModel):
    token: str = Field(min_length=32, max_length=32)
    dataset_id: str = Field(min_length=2, max_length=63)
    name: str = Field("", max_length=120)
    license: str = Field("未声明", max_length=200)
    citation: str = Field("", max_length=500)
    description: str = Field("", max_length=1000)
    overwrite: bool = False
    channels: list[str] | None = None
    event_map: dict[str, str] | None = None
    label_column: str | None = None
    time_column: str | None = None
    trial_column: str | None = None
    subject_column: str | None = None
    session_column: str | None = None
    sfreq: float | None = Field(None, gt=0, le=100000)
    tmin: float = 0.0
    tmax: float = 1.0
    baseline: list[float] | None = None
    resample: float | None = Field(None, gt=0, le=4096)
    paradigm: str = "MI"
    subject_id: str = Field("S01", max_length=64)
    session_id: str = Field("0", max_length=64)
    positive_label: str | None = Field(None, max_length=64)
    stimulus_frequencies: list[float] | None = None


def _load_epochs(req: DatasetRequest):
    return registry.load(
        req.dataset_id,
        subject_ids=req.subject_ids or None,
        classes=req.classes,
        options=req.options,
    )


def _manifest(req: DatasetRequest, epochs) -> dict[str, Any]:
    manifest = registry.manifest(req.dataset_id, epochs)
    manifest["dataset_version"] = epochs.dataset_version
    manifest["selection"] = {
        "subject_ids": req.subject_ids,
        "classes": req.classes,
        "options": req.options,
    }
    return manifest


def _dataset_public(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "name", "paradigm", "paradigm_meta", "license", "citation", "dataset_version",
        "data_signature", "loaded_n_trials", "loaded_channels", "loaded_sfreq",
        "loaded_classes", "label_names", "subjects", "sessions", "tmin", "tmax",
    )
    return {key: manifest.get(key) for key in keys}


def _public_rows(rows: list[dict]) -> list[dict]:
    public = []
    for row in rows:
        public.append(
            {
                **row,
                "splits": [
                    {
                        key: value for key, value in split.items()
                        if key not in ("y_true", "y_pred", "y_proba", "proba_classes")
                    }
                    for split in row.get("splits", [])
                ],
            }
        )
    return public


def _execute_arena(
    req: ArenaRequest,
    on_progress: Callable[[int, int, dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    result = run_experiment(
        registry,
        store,
        ExperimentRequest(
            dataset_id=req.dataset_id,
            subject_ids=list(req.subject_ids),
            classes=req.classes,
            options=dict(req.options),
            protocol=req.protocol,
            folds=req.folds,
            repeats=req.repeats,
            seeds=list(req.seeds) or [0],
            pipelines=list(req.pipelines),
            name=req.name,
            notes=req.notes,
            save=req.save,
            use_cache=req.use_cache,
        ),
        on_progress=on_progress,
        should_cancel=should_cancel,
        plugins=plugins,
    )
    response = {key: value for key, value in result.items() if key != "manifest"}
    response["rows"] = _public_rows(result["rows"])
    return response


def _current_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return user


def _require_admin(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


@app.get("/api/health")
def health():
    stats = store.stats()
    return {
        "status": "ok",
        "api": app.version,
        "engine": engine_version,
        "engine_path": str(ROOT / "engine"),
        "data_root": str(DATA_ROOT),
        "database": str(store.path),
        "moabb_installed": registry.moabb_installed,
        "torch_available": torch_available(),
        "web_ui": WEB_DIST.is_dir(),
        "datasets_available": len(registry.specs()),
        "experiments": stats["experiments"],
        "custom_pipelines": stats["custom_pipelines"],
        "auth_initialized": not auth.setup_required(),
        "leakage_guard": True,
    }


@app.get("/api/cache")
def cache_stats():
    return registry.artifact_cache.stats()


@app.delete("/api/cache")
def cache_clear(_user: dict[str, Any] = Depends(_require_admin)):
    return registry.artifact_cache.clear()


@app.get("/api/auth/status")
def auth_status():
    return {
        "initialized": not auth.setup_required(),
        "user_count": auth.user_count(),
        "auth_required": True,
    }


@app.post("/api/auth/bootstrap")
def auth_bootstrap(credentials: AuthCredentials):
    try:
        user = auth.bootstrap(credentials.username, credentials.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return auth.issue_token(user)


@app.post("/api/auth/login")
def auth_login(credentials: AuthCredentials):
    user = auth.authenticate(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return auth.issue_token(user)


@app.get("/api/auth/me")
def auth_me(user: dict[str, Any] = Depends(_current_user)):
    fresh = auth.get_user(int(user["sub"]))
    if fresh is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return fresh


@app.get("/api/auth/users")
def auth_users(_user: dict[str, Any] = Depends(_require_admin)):
    items = auth.list_users()
    return {"items": items, "total": len(items)}


@app.post("/api/auth/users")
def auth_create_user(
    payload: UserCreate,
    _user: dict[str, Any] = Depends(_require_admin),
):
    try:
        return auth.create_user(payload.username, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/auth/users/{user_id}")
def auth_delete_user(
    user_id: int,
    user: dict[str, Any] = Depends(_require_admin),
):
    if int(user["sub"]) == user_id:
        raise HTTPException(status_code=400, detail="不能删除当前登录账号")
    try:
        if not auth.delete_user(user_id):
            raise HTTPException(status_code=404, detail="用户不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True, "id": user_id}


@app.get("/api/datasets")
def datasets():
    items = registry.describe()
    return {"items": items, "total": len(items)}


@app.post("/api/datasets/{dataset_id}/load")
def load_dataset(dataset_id: str, req: Selection):
    try:
        epochs = registry.load(
            dataset_id,
            subject_ids=req.subject_ids or None,
            classes=req.classes,
            options=req.options,
        )
        manifest = _manifest(
            DatasetRequest(
                dataset_id=dataset_id, subject_ids=req.subject_ids,
                classes=req.classes, options=req.options,
            ),
            epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"数据集加载失败: {exc}") from exc
    metadata = manifest["selection"]
    trials = [
        {
            "trial_id": str(epochs.trial_ids[index]),
            "subject": str(epochs.subject_ids[index]),
            "session": str(epochs.session_ids[index]),
            "label": int(epochs.labels[index]),
            "label_name": (
                epochs.label_names[int(epochs.labels[index])]
                if int(epochs.labels[index]) < len(epochs.label_names)
                else str(int(epochs.labels[index]))
            ),
        }
        for index in range(min(12, epochs.n_trials))
    ]
    return {
        "dataset": _dataset_public(manifest),
        "manifest": manifest,
        "selection": metadata,
        "shape": {
            "trials": epochs.n_trials,
            "channels": len(epochs.channel_names),
            "times": int(epochs.data.shape[2]),
        },
        "channel_names": list(epochs.channel_names),
        "label_names": list(epochs.label_names),
        "label_counts": {str(key): value for key, value in epochs.label_counts().items()},
        "subject_ids": epochs.subjects,
        "session_ids": sorted(set(epochs.session_ids.tolist())),
        "trials": trials,
    }


@app.post("/api/datasets/import")
def import_dataset(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    name: str = Form(""),
    license: str = Form("未声明"),
    citation: str = Form(""),
    description: str = Form(""),
    overwrite: bool = Form(False),
    user: dict[str, Any] = Depends(_current_user),
):
    if overwrite and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="覆盖已有数据集需要管理员权限")
    filename = (file.filename or "").lower()
    if not filename.endswith(".npz"):
        raise HTTPException(status_code=400, detail="仅支持 .npz 文件")
    import_dir = DATA_ROOT / "imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(suffix=".npz", dir=import_dir)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as target:
            shutil.copyfileobj(file.file, target)
            size = target.tell()
        if size > 2 * 1024**3:
            raise HTTPException(status_code=413, detail="文件超过 2 GiB 上限")
        spec = registry.import_npz(
            temp_path,
            dataset_id=dataset_id,
            name=name or dataset_id,
            license=license,
            citation=citation,
            description=description,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return {
        "item": registry.describe(spec.id)[0],
        "spec": spec.to_dict(),
    }


IMPORT_STAGE = DATA_ROOT / "imports" / "_stage"
IMPORT_STAGE_TOKEN = re.compile(r"^[a-f0-9]{32}$")
IMPORT_SUFFIXES = {".npz", ".edf", ".fif", ".csv"}
MAX_IMPORT_BYTES = 2 * 1024**3


def _cleanup_import_stage(max_age_seconds: int = 24 * 3600) -> None:
    cutoff = time.time() - max_age_seconds
    for path in IMPORT_STAGE.glob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _stage_file(token: str) -> Path:
    if not IMPORT_STAGE_TOKEN.fullmatch(token):
        raise HTTPException(status_code=400, detail="非法的导入令牌")
    matches = [path for path in IMPORT_STAGE.glob(f"{token}.*") if path.is_file()]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="导入文件已过期，请重新上传")
    return matches[0]


def _import_config(req: DatasetImportCommit) -> dict[str, Any]:
    return {
        "channels": req.channels,
        "event_map": req.event_map,
        "label_column": req.label_column,
        "time_column": req.time_column,
        "trial_column": req.trial_column,
        "subject_column": req.subject_column,
        "session_column": req.session_column,
        "sfreq": req.sfreq,
        "tmin": req.tmin,
        "tmax": req.tmax,
        "baseline": req.baseline,
        "resample": req.resample,
        "paradigm": req.paradigm,
        "subject_id": req.subject_id,
        "session_id": req.session_id,
        "positive_label": req.positive_label,
        "stimulus_frequencies": req.stimulus_frequencies,
    }


@app.post("/api/datasets/import/inspect")
def inspect_dataset_import(file: UploadFile = File(...)):
    filename = (file.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in IMPORT_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="仅支持 .npz / .edf / .fif / .csv 文件",
        )
    IMPORT_STAGE.mkdir(parents=True, exist_ok=True)
    _cleanup_import_stage()
    token = uuid.uuid4().hex
    stage_path = IMPORT_STAGE / f"{token}{suffix}"
    size = 0
    try:
        with stage_path.open("wb") as target:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_IMPORT_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 2 GiB 上限")
                target.write(chunk)
        info = inspect_import_file(stage_path)
    except HTTPException:
        stage_path.unlink(missing_ok=True)
        raise
    except (ValueError, OSError) as exc:
        stage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # MNE may raise parser-specific exceptions
        stage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"文件解析失败：{exc}") from exc
    return {
        "token": token,
        "filename": filename,
        "size": size,
        **info,
    }


@app.post("/api/datasets/import/commit")
def commit_dataset_import(req: DatasetImportCommit, user: dict[str, Any] = Depends(_current_user)):
    if req.overwrite and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="覆盖已有数据集需要管理员权限")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,62}", req.dataset_id):
        raise HTTPException(status_code=400, detail="数据集 ID 需匹配 [a-z0-9][a-z0-9_-]{1,62}")
    if req.tmax <= req.tmin:
        raise HTTPException(status_code=400, detail="tmax 必须大于 tmin")
    if req.tmax - req.tmin > 60:
        raise HTTPException(status_code=400, detail="单个试次窗口不能超过 60 秒")
    if req.baseline is not None:
        if len(req.baseline) != 2 or float(req.baseline[0]) >= float(req.baseline[1]):
            raise HTTPException(status_code=400, detail="baseline 必须为 [起始, 结束] 且起始小于结束")
    if req.channels is not None and len(req.channels) > 512:
        raise HTTPException(status_code=400, detail="通道数量超过 512 上限")
    stage_path = _stage_file(req.token)
    suffix = stage_path.suffix.lower()
    converted_path = IMPORT_STAGE / f"{req.token}-converted.npz"
    epochs = None
    try:
        if suffix == ".npz":
            spec = registry.import_npz(
                stage_path,
                dataset_id=req.dataset_id,
                name=req.name or req.dataset_id,
                license=req.license,
                citation=req.citation,
                description=req.description,
                overwrite=req.overwrite,
            )
        elif suffix == ".csv":
            epochs = build_csv_epochs(stage_path, _import_config(req), req.dataset_id)
            write_epochs_npz(epochs, converted_path)
            spec = registry.import_npz(
                converted_path,
                dataset_id=req.dataset_id,
                name=req.name or req.dataset_id,
                license=req.license,
                citation=req.citation,
                description=req.description,
                overwrite=req.overwrite,
            )
        elif suffix in {".edf", ".fif"}:
            epochs = build_raw_epochs(stage_path, _import_config(req), req.dataset_id)
            write_epochs_npz(epochs, converted_path)
            spec = registry.import_npz(
                converted_path,
                dataset_id=req.dataset_id,
                name=req.name or req.dataset_id,
                license=req.license,
                citation=req.citation,
                description=req.description,
                overwrite=req.overwrite,
            )
        else:  # pragma: no cover - inspect already rejects unknown suffixes
            raise ValueError("不支持的导入格式")
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=f"转换失败：{exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"文件读写失败：{exc}") from exc
    finally:
        converted_path.unlink(missing_ok=True)
    stage_path.unlink(missing_ok=True)
    summary = None
    if epochs is not None:
        summary = {
            "n_trials": epochs.n_trials,
            "n_channels": len(epochs.channel_names),
            "sfreq": epochs.sfreq,
            "classes": epochs.n_classes,
            "label_names": list(epochs.label_names),
            "subjects": epochs.subjects,
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }
    return {
        "item": registry.describe(spec.id)[0],
        "spec": spec.to_dict(),
        "summary": summary,
    }


@app.get("/api/pipelines")
def pipelines():
    items = [{**item, "paradigm": item.get("paradigm") or "MI"} for item in PIPELINE_DEFINITIONS]
    items.extend(plugins.definitions())
    return {"items": items, "total": len(items)}


@app.get("/api/plugins")
def plugin_status():
    return plugins.status()


@app.post("/api/plugins/install/bundled")
def install_bundled_plugin(
    payload: PluginInstallRequest,
    _user: dict[str, Any] = Depends(_require_admin),
):
    try:
        return plugins.install_bundled(payload.id, enable=payload.enable)
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plugins/install")
async def install_plugin_zip(
    file: UploadFile = File(...),
    enable: bool = Form(True),
    _user: dict[str, Any] = Depends(_require_admin),
):
    filename = file.filename or "plugin.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 .zip 插件包")
    payload = await file.read(60 * 1024 * 1024 + 1)
    if len(payload) > 60 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="插件包超过 60 MiB")
    try:
        return plugins.install_zip(payload, filename=filename, enable=enable)
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plugins/{plugin_id}/enable")
def enable_plugin(
    plugin_id: str,
    _user: dict[str, Any] = Depends(_require_admin),
):
    try:
        return plugins.enable(plugin_id)
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plugins/{plugin_id}/disable")
def disable_plugin(
    plugin_id: str,
    _user: dict[str, Any] = Depends(_require_admin),
):
    try:
        return plugins.disable(plugin_id)
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/plugins/{plugin_id}")
def uninstall_plugin(
    plugin_id: str,
    _user: dict[str, Any] = Depends(_require_admin),
):
    try:
        plugins.uninstall(plugin_id)
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True, "id": plugin_id}


@app.get("/api/pipelines/catalog")
def pipeline_catalog():
    items = node_catalog()
    return {"items": items, "total": len(items)}


@app.get("/api/pipelines/templates")
def pipeline_templates():
    items = pipeline_template_catalog()
    for item in items:
        item["search_space_count"] = len(item.get("search_space") or [])
        item["node_count"] = len(item.get("spec", {}).get("nodes") or [])
    return {"items": items, "total": len(items)}


@app.get("/api/reproduction")
def reproduction_index():
    items = reproduction_catalog()
    return {"items": items, "total": len(items)}


@app.post("/api/pipelines/templates/{template_id}/install")
def install_pipeline_template(template_id: str, payload: TemplateInstallRequest):
    try:
        template = get_pipeline_template(template_id)
        spec = normalize_graph(template["spec"])
        dims = (
            normalize_search_space(spec, template["search_space"])
            if template.get("search_space")
            else []
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (GraphValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    paper = template.get("paper") or {}
    source = ""
    if paper:
        source = (
            f" 论文：{paper.get('authors', '')} ({paper.get('year', '')}). "
            f"{paper.get('title', '')}"
        )
        if paper.get("doi"):
            source += f" doi:{paper['doi']}"
    pipeline_id = store.save_custom_pipeline(
        payload.name.strip() or template["name"],
        f"{template['description']}{source}".strip()[:500],
        spec,
    )
    if dims:
        store.save_custom_pipeline_space(pipeline_id, dims)
    item = store.get_custom_pipeline(pipeline_id)
    item["paradigm"] = infer_spec_paradigm(item.get("spec") or {})
    return {"item": item, "template_id": template_id, "search_space_count": len(dims)}


@app.get("/api/pipelines/custom/{pipeline_id}/export")
def export_custom_pipeline(pipeline_id: int):
    item = store.get_custom_pipeline(pipeline_id)
    if item is None:
        raise HTTPException(status_code=404, detail="自定义算法不存在")
    try:
        spec = normalize_graph(item["spec"])
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dims = store.get_custom_pipeline_space(pipeline_id) or []
    payload = {
        "format": TEMPLATE_FORMAT,
        "version": TEMPLATE_FORMAT_VERSION,
        "name": item["name"],
        "description": item.get("description") or "",
        "spec": spec,
        "search_space": dims,
        "template": {
            "source": "export",
            "pipeline_id": pipeline_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["checksum"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


@app.post("/api/pipelines/custom/import")
def import_custom_pipeline(payload: PipelineTemplateImport):
    if payload.format != TEMPLATE_FORMAT:
        raise HTTPException(status_code=400, detail="不是 NeuroKiln 丹方模板文件")
    if payload.version > TEMPLATE_FORMAT_VERSION:
        raise HTTPException(status_code=400, detail="模板版本高于当前支持版本")
    if payload.checksum:
        body = payload.model_dump(exclude={"checksum"})
        canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(expected, payload.checksum):
            raise HTTPException(status_code=400, detail="模板校验和不匹配，文件可能被修改")
    try:
        spec = normalize_graph(payload.spec)
        dims = (
            normalize_search_space(spec, payload.search_space)
            if payload.search_space
            else []
        )
    except (GraphValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    name = payload.name.strip() or "导入的丹方"
    pipeline_id = store.save_custom_pipeline(name, payload.description[:500], spec)
    if dims:
        store.save_custom_pipeline_space(pipeline_id, dims)
    item = store.get_custom_pipeline(pipeline_id)
    item["paradigm"] = infer_spec_paradigm(item.get("spec") or {})
    return {"item": item, "search_space_count": len(dims)}


@app.get("/api/pipelines/custom")
def list_custom_pipelines():
    items = store.list_custom_pipelines()
    for item in items:
        item["paradigm"] = infer_spec_paradigm(item.get("spec") or {})
    return {"items": items, "total": len(items)}


@app.post("/api/pipelines/custom")
def create_custom_pipeline(payload: CustomPipelineRequest):
    try:
        spec = normalize_graph(payload.spec)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pipeline_id = store.save_custom_pipeline(payload.name, payload.description, spec)
    return store.get_custom_pipeline(pipeline_id)


@app.get("/api/pipelines/custom/{pipeline_id}")
def get_custom_pipeline(pipeline_id: int):
    item = store.get_custom_pipeline(pipeline_id)
    if item is None:
        raise HTTPException(status_code=404, detail="自定义算法不存在")
    return item


@app.put("/api/pipelines/custom/{pipeline_id}")
def update_custom_pipeline(pipeline_id: int, payload: CustomPipelineRequest):
    try:
        spec = normalize_graph(payload.spec)
        store.save_custom_pipeline(
            payload.name, payload.description, spec, pipeline_id=pipeline_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="自定义算法不存在") from exc
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.get_custom_pipeline(pipeline_id)


@app.delete("/api/pipelines/custom/{pipeline_id}")
def delete_custom_pipeline(pipeline_id: int, _user: dict[str, Any] = Depends(_require_admin)):
    if not store.delete_custom_pipeline(pipeline_id):
        raise HTTPException(status_code=404, detail="自定义算法不存在")
    return {"deleted": True, "id": pipeline_id}


@app.post("/api/pipelines/validate")
def validate_custom_pipeline(payload: CustomPipelineRequest):
    try:
        spec = normalize_graph(payload.spec)
    except GraphValidationError as exc:
        return {"valid": False, "error": str(exc)}
    return {
        "valid": True,
        "topology": spec["topology"],
        "decoder": spec["decoder"],
        "decoders": spec["decoders"],
        "prediction": spec["prediction"],
        "nodes": len(spec["nodes"]),
        "edges": len(spec["edges"]),
    }


@app.get("/api/pipelines/custom/{pipeline_id}/search-space")
def get_custom_pipeline_space(pipeline_id: int):
    item = store.get_custom_pipeline(pipeline_id)
    if item is None:
        raise HTTPException(status_code=404, detail="自定义算法不存在")
    dims = store.get_custom_pipeline_space(pipeline_id) or []
    return {"pipeline_id": pipeline_id, "dims": dims, "total": len(dims)}


@app.put("/api/pipelines/custom/{pipeline_id}/search-space")
def save_custom_pipeline_space(pipeline_id: int, payload: SearchSpaceRequest):
    item = store.get_custom_pipeline(pipeline_id)
    if item is None:
        raise HTTPException(status_code=404, detail="自定义算法不存在")
    if not payload.dims:
        store.delete_custom_pipeline_space(pipeline_id)
        return {"pipeline_id": pipeline_id, "dims": [], "total": 0}
    try:
        dims = normalize_search_space(item["spec"], payload.dims)
    except (GraphValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save_custom_pipeline_space(pipeline_id, dims)
    return {"pipeline_id": pipeline_id, "dims": dims, "total": len(dims)}


@app.get("/api/codex")
def codex():
    return {"items": CODEX, "total": len(CODEX)}


def _custom_template_entry(
    pipeline: dict[str, Any], dims: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        template = custom_template(
            int(pipeline["id"]),
            str(pipeline["name"]),
            str(pipeline.get("description") or ""),
            pipeline["spec"],
            dims,
        )
    except (GraphValidationError, ValueError, KeyError, TypeError) as exc:
        return {
            "id": f"custom:{pipeline['id']}",
            "name": pipeline["name"],
            "description": pipeline.get("description") or "",
            "family": "custom",
            "deep": False,
            "space": [],
            "defaults": {},
            "custom_pipeline_id": int(pipeline["id"]),
            "valid": False,
            "error": str(exc),
        }
    entry = template.describe()
    entry["custom_pipeline_id"] = int(pipeline["id"])
    entry["valid"] = True
    return entry


def _resolve_hpo_template(template_id: str):
    if not str(template_id).startswith("custom:"):
        return None
    suffix = str(template_id).split(":", 1)[1]
    try:
        pipeline_id = int(suffix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="自定义模板 ID 非法") from exc
    pipeline = store.get_custom_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=400, detail="自定义丹方不存在")
    dims = store.get_custom_pipeline_space(pipeline_id)
    if not dims:
        raise HTTPException(
            status_code=400,
            detail="该自定义丹方还没有配置搜索空间；请先在丹方设计页设置可调参数",
        )
    try:
        return custom_template(
            pipeline_id,
            str(pipeline["name"]),
            str(pipeline.get("description") or ""),
            pipeline["spec"],
            dims,
        )
    except (GraphValidationError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/hpo/templates")
def hpo_templates():
    items = template_catalog()
    spaces = store.list_custom_pipeline_spaces()
    if spaces:
        pipelines = {item["id"]: item for item in store.list_custom_pipelines()}
        for pipeline_id in sorted(spaces):
            pipeline = pipelines.get(pipeline_id)
            if pipeline is None:
                continue
            items.append(_custom_template_entry(pipeline, spaces[pipeline_id]))
    return {
        "items": items,
        "objectives": objective_catalog(),
        "total": len(items),
    }


@app.get("/api/protocols")
def protocols():
    return {"items": PROTOCOLS, "total": len(PROTOCOLS)}


@app.post("/api/protocols/preview")
def protocol_preview(req: ProtocolRequest):
    try:
        epochs = _load_epochs(req)
        manifest = _manifest(req, epochs)
        splits = build_protocol(
            req.protocol, folds=req.folds, repeats=req.repeats, seeds=tuple(req.seeds)
        ).split(epochs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"数据集加载失败: {exc}") from exc
    return {
        "dataset": _dataset_public(manifest),
        "n_splits": len(splits),
        "splits": [split.summary(epochs) for split in splits],
    }


@app.post("/api/arena")
def arena(req: ArenaRequest):
    try:
        return _execute_arena(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"数据集加载失败: {exc}") from exc


@app.post("/api/arena/demo")
def arena_demo(req: ArenaRequest):
    return arena(req)


@app.post("/api/demo/run")
def golden_demo():
    req = ArenaRequest(
        dataset_id="synthetic",
        subject_ids=[],
        options={
            "n_trials_per_class": 20,
            "n_times": 48,
            "class_sep": 0.02,
            "class_noise_step": 0.2,
        },
        protocol="within_subject",
        folds=4,
        repeats=1,
        seeds=[0],
        pipelines=["csp_lda", "riemann_mdm"],
        name="黄金演示 · 5 分钟上手",
        notes="内置合成 EEG 数据，仅用于验证评测流程，不代表真实 BCI 性能。",
        save=True,
        use_cache=True,
    )
    try:
        result = _execute_arena(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["demo"] = {
        "kind": "golden",
        "dataset_notice": "合成数据仅用于演示流程；真实结论请导入 EDF/FIF/CSV 或使用 MOABB 数据集。",
        "steps": [
            "载入内置合成 EEG",
            "CSP+LDA 与 Riemann MDM 各跑 4 折被试内评测",
            "自动保存实验并跳转报告",
        ],
    }
    return result


def _run_arena_job(req: ArenaRequest):
    def runner(job_id, on_progress, should_cancel):
        def report(done: int, total: int, detail: dict[str, Any]) -> None:
            on_progress(
                done / max(total, 1),
                f"{detail.get('pipeline', '')} · 第 {detail.get('split_index', 0) + 1}/"
                f"{detail.get('n_splits', 0)} 折",
            )

        try:
            return _execute_arena(
                req, on_progress=report, should_cancel=should_cancel
            )
        except RunCancelled as exc:
            raise JobCancelled(str(exc)) from exc

    return runner


@app.post("/api/jobs/arena")
def submit_arena_job(req: ArenaRequest):
    job = jobs.submit("arena", req.model_dump(), _run_arena_job(req))
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "message": job.get("message", ""),
    }


def _run_hpo_job(req: HPOJobRequest, template=None):
    def runner(job_id, on_progress, should_cancel):
        def report(done: int, total: int, info: dict[str, Any]) -> None:
            stage = "完整复评" if info.get("stage") == "full" else "代理筛选"
            message = f"{stage} {done}/{total}"
            metrics = info.get("metrics") or {}
            if metrics.get("balanced_accuracy") is not None:
                message += f" · BA {float(metrics['balanced_accuracy']):.4f}"
            if info.get("error"):
                message += " · 该候选失败"
            on_progress(done / max(total, 1), message)

        try:
            return run_hpo(
                registry,
                HPORequest(
                    dataset_id=req.dataset_id,
                    subject_ids=list(req.subject_ids),
                    classes=req.classes,
                    options=dict(req.options),
                    protocol=req.protocol,
                    folds=req.folds,
                    repeats=req.repeats,
                    seeds=list(req.seeds) or [0],
                    template_id=req.template_id,
                    objectives=list(req.objectives),
                    budget=req.budget,
                    population=req.population,
                    proxy_folds=req.proxy_folds,
                    full_candidates=req.full_candidates,
                    seed=req.seed,
                    name=req.name,
                    notes=req.notes,
                ),
                on_progress=report,
                should_cancel=should_cancel,
                template=template,
            )
        except RunCancelled as exc:
            raise JobCancelled(str(exc)) from exc

    return runner


@app.post("/api/hpo/jobs")
def submit_hpo_job(req: HPOJobRequest):
    template = _resolve_hpo_template(req.template_id)
    job = jobs.submit("hpo", req.model_dump(), _run_hpo_job(req, template))
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "message": job.get("message", ""),
    }


@app.post("/api/hpo/promote")
def promote_hpo_result(req: HPOPromoteRequest):
    job = jobs.get(req.job_id)
    if job is None or job.get("kind") != "hpo":
        raise HTTPException(status_code=404, detail="HPO 任务不存在")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="HPO 任务尚未完成")
    best = (job.get("result") or {}).get("best") or {}
    graph = best.get("graph")
    if not graph:
        raise HTTPException(status_code=400, detail="该任务没有可用的推荐丹方")
    try:
        spec = normalize_graph(graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    definition = best.get("definition") or {}
    name = req.name.strip() or str(definition.get("name") or "HPO 推荐丹方")
    description = req.description.strip() or str(definition.get("note") or "")
    pipeline_id = store.save_custom_pipeline(name, description, spec)
    return store.get_custom_pipeline(pipeline_id)


@app.get("/api/jobs")
def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    items, total = jobs.list(limit=limit, offset=offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    return {"cancel_requested": True, "id": job_id}


@app.get("/api/experiments")
def list_experiments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    dataset_id: str | None = None,
    status: str | None = None,
):
    items, total = store.list_experiments(
        limit=limit, offset=offset, dataset_id=dataset_id, status=status
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/experiments/stats")
def experiment_stats():
    stats = store.stats()
    specs = registry.specs()
    stats.update(
        {
            "datasets_available": len(specs),
            "local_datasets": sum(1 for item in specs if item.adapter == "local_npz"),
            "moabb_datasets": sum(1 for item in specs if item.adapter == "moabb"),
            "moabb_installed": registry.moabb_installed,
        }
    )
    return stats


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: int):
    detail = store.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return detail


@app.get("/api/experiments/{experiment_id}/report.html")
def experiment_report_html(experiment_id: int):
    detail = store.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return Response(
        content=build_html_report(detail),
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="neurokiln-experiment-{experiment_id}.html"'
        },
    )


@app.get("/api/experiments/{experiment_id}/report.xlsx")
def experiment_report_xlsx(experiment_id: int):
    detail = store.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return Response(
        content=build_workbook(detail),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="neurokiln-experiment-{experiment_id}.xlsx"'
        },
    )


@app.get("/api/experiments/{experiment_id}/report.docx")
def experiment_report_docx(experiment_id: int):
    detail = store.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return Response(
        content=build_docx(detail),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="neurokiln-experiment-{experiment_id}.docx"'
        },
    )


@app.get("/api/experiments/{experiment_id}/report.tex")
def experiment_report_latex(experiment_id: int):
    detail = store.get_experiment(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return Response(
        content=build_latex(detail),
        media_type="application/x-tex; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="neurokiln-experiment-{experiment_id}.tex"'
        },
    )


@app.delete("/api/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, _user: dict[str, Any] = Depends(_require_admin)):
    if not store.delete_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="实验不存在")
    return {"deleted": True, "id": experiment_id}


@app.on_event("shutdown")
def shutdown_jobs():
    jobs.shutdown()


if WEB_DIST.is_dir():
    assets_dir = WEB_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "docs", "redoc", "openapi.json")):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (WEB_DIST / full_path).resolve()
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if candidate.is_file() and candidate.is_relative_to(WEB_DIST):
            headers = (
                {"Cache-Control": "no-cache"}
                if candidate == WEB_DIST / "index.html"
                else None
            )
            return FileResponse(candidate, headers=headers)
        return FileResponse(
            WEB_DIST / "index.html",
            headers={"Cache-Control": "no-cache"},
        )
