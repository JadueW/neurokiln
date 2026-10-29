"""Local plugin marketplace with hash-verified, permission-scoped packages.

Graph plugins are declarative DAG specs and never execute third-party code.
Python plugins are accepted only in trusted mode (``NEUROKILN_PLUGIN_TRUST``)
and expose additional DAG nodes through ``register(api)``. Python code runs
in-process and cannot be sandboxed; enable trusted mode only for packages you
have reviewed. Every activation re-verifies the manifest and file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import types
import uuid
import zipfile
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from . import __version__ as ENGINE_VERSION
from .graph import (
    GraphValidationError,
    build_graph_pipeline,
    normalize_graph,
    register_plugin_node_specs,
    unregister_plugin_node_specs,
)

PLUGIN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
_PIPELINE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?$")
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][A-Za-z0-9.]+)?$")
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
NODE_API_VERSION = 1
_ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt"}
_MAX_FILES = 200
_MAX_FILE_BYTES = 20 * 1024 * 1024
_MAX_TOTAL_BYTES = 60 * 1024 * 1024
_PARADIGMS = {"MI", "P300", "SSVEP"}
_RUNTIME_PERMISSIONS = {"graph": "graph", "python": "nodes"}
_COMPUTED_FIELDS = {"manifest_sha256", "package_sha256", "installed_at", "source"}


class PluginError(ValueError):
    pass


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = _SEMVER.match(str(value).strip())
    if match is None:
        raise PluginError(f"版本号必须形如 1.2.3: {value!r}")
    return tuple(int(match.group(index)) for index in (1, 2, 3))


def plugin_compatibility(
    manifest: dict[str, Any], engine_version: str = ENGINE_VERSION
) -> tuple[bool, str]:
    """Check manifest schema and engine/API compatibility."""
    schema_version = int(manifest.get("schema_version", 1))
    if schema_version < SCHEMA_VERSION:
        return True, "旧版清单 v1（兼容模式）"
    required = manifest.get("engine_min_version") or "0.0.0"
    try:
        required_version = _version_tuple(required)
        current_version = _version_tuple(engine_version)
    except PluginError as exc:
        return False, str(exc)
    if current_version < required_version:
        return False, f"需要引擎 >= {required}，当前 {engine_version}"
    api_version = int(manifest.get("api_version") or 1)
    if api_version > NODE_API_VERSION:
        return False, f"需要节点 API v{api_version}，当前支持 v{NODE_API_VERSION}"
    return True, "兼容"


def _ensure_compatible(manifest: dict[str, Any]) -> None:
    compatible, reason = plugin_compatibility(manifest)
    if not compatible:
        raise PluginError(f"插件 {manifest.get('id', '')} 不兼容: {reason}")


def _compat_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    compatible, reason = plugin_compatibility(manifest)
    return {
        "schema_version": int(manifest.get("schema_version", 1)),
        "engine_min_version": manifest.get("engine_min_version", ""),
        "api_version": int(manifest.get("api_version", 1)),
        "compatible": compatible,
        "compatibility": reason,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key not in _COMPUTED_FIELDS}
    return _sha256(_canonical(payload))


def _package_digest(files: dict[str, str]) -> str:
    return _sha256(_canonical(files))


def _strip_single_root(files: dict[str, bytes]) -> dict[str, bytes]:
    if "plugin.json" in files:
        return files
    prefixes = {name.split("/", 1)[0] for name in files if "/" in name}
    if len(prefixes) == 1:
        prefix = next(iter(prefixes)) + "/"
        return {
            name[len(prefix):]: payload
            for name, payload in files.items()
            if name.startswith(prefix)
        }
    return files


def _read_zip_files(payload: bytes, filename: str = "plugin.zip") -> dict[str, bytes]:
    if len(payload) > _MAX_TOTAL_BYTES:
        raise PluginError("插件 ZIP 超过 60 MiB")
    try:
        archive = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise PluginError(f"不是合法的 ZIP 文件: {filename}") from exc
    files: dict[str, bytes] = {}
    total = 0
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > _MAX_FILES:
            raise PluginError(f"ZIP 文件数不能超过 {_MAX_FILES}")
        for info in infos:
            name = _safe_relative(info.filename)
            if info.file_size > _MAX_FILE_BYTES:
                raise PluginError(f"文件 {name} 超过 20 MiB")
            total += info.file_size
            if total > _MAX_TOTAL_BYTES:
                raise PluginError("ZIP 解压后超过 60 MiB")
            files[name] = archive.read(info)
    return files


def _safe_relative(name: str) -> str:
    cleaned = str(name).replace("\\", "/").strip()
    path = PurePosixPath(cleaned)
    if not cleaned or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PluginError(f"插件包含非法路径: {name!r}")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise PluginError(f"插件包含不支持的文件类型: {name}")
    return path.as_posix()


def _require_text(raw: dict[str, Any], key: str, max_length: int, default: str = "") -> str:
    value = str(raw.get(key) or default).strip()
    if len(value) > max_length:
        raise PluginError(f"字段 {key} 超过 {max_length} 个字符")
    return value


def normalize_manifest(raw: dict[str, Any], plugin_id: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PluginError("plugin.json 必须是对象")
    try:
        schema_version = int(raw.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise PluginError("schema_version 必须是整数 1 或 2") from exc
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PluginError(
            f"不支持的插件清单版本: {schema_version}；"
            f"当前支持 {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    engine_min_version = ""
    api_version = 1
    if schema_version >= 2:
        engine_min_version = _require_text(raw, "engine_min_version", 32)
        if not engine_min_version:
            raise PluginError("清单 v2 必须声明 engine_min_version，例如 0.1.0")
        _version_tuple(engine_min_version)
        api_raw = raw.get("api_version")
        if api_raw is None:
            raise PluginError("清单 v2 必须声明 api_version（当前为 1）")
        if isinstance(api_raw, bool):
            raise PluginError("api_version 必须是正整数")
        try:
            api_version = int(api_raw)
        except (TypeError, ValueError) as exc:
            raise PluginError("api_version 必须是正整数") from exc
        if api_version < 1:
            raise PluginError("api_version 必须是正整数")
    identifier = _require_text(raw, "id", 41)
    if not PLUGIN_ID.match(identifier):
        raise PluginError("插件 ID 只能包含小写字母、数字、- 和 _，长度 2–41")
    if plugin_id is not None and identifier != plugin_id:
        raise PluginError(f"插件目录与清单 ID 不一致: {plugin_id} != {identifier}")
    name = _require_text(raw, "name", 80)
    if not name:
        raise PluginError("插件缺少 name")
    version = _require_text(raw, "version", 32)
    if not _VERSION.match(version):
        raise PluginError("插件 version 必须形如 1.0.0")
    runtime = _require_text(raw, "runtime", 16, "graph").lower()
    if runtime not in _RUNTIME_PERMISSIONS:
        raise PluginError("runtime 只能是 graph 或 python")
    permissions = [str(item).strip() for item in raw.get("permissions") or []]
    if not permissions:
        permissions = [_RUNTIME_PERMISSIONS[runtime]]
    if len(set(permissions)) != len(permissions):
        raise PluginError("permissions 不能重复")
    required = _RUNTIME_PERMISSIONS[runtime]
    if set(permissions) != {required}:
        raise PluginError(
            f"{runtime} 插件的权限必须恰好为 ['{required}']；"
            "插件系统不支持网络、文件系统或子进程权限"
        )
    entry = _require_text(raw, "entry", 160)
    if runtime == "python":
        if not entry.lower().endswith(".py"):
            raise PluginError("Python 插件必须在 entry 中声明 .py 入口")
        _safe_relative(entry)
    elif entry:
        raise PluginError("声明式图插件不需要 entry")

    raw_pipelines = raw.get("pipelines")
    if not isinstance(raw_pipelines, list) or not raw_pipelines:
        raise PluginError("插件至少需要一条 pipeline")
    if len(raw_pipelines) > 12:
        raise PluginError("单个插件最多提供 12 条 pipeline")
    pipelines = []
    seen = set()
    for item in raw_pipelines:
        if not isinstance(item, dict):
            raise PluginError("pipeline 定义必须是对象")
        pipeline_id = _require_text(item, "id", 41)
        if not _PIPELINE_ID.match(pipeline_id):
            raise PluginError(f"pipeline ID 非法: {pipeline_id!r}")
        if pipeline_id in seen:
            raise PluginError(f"pipeline ID 重复: {pipeline_id}")
        seen.add(pipeline_id)
        pipeline_name = _require_text(item, "name", 80)
        if not pipeline_name:
            raise PluginError(f"pipeline {pipeline_id} 缺少 name")
        paradigm = _require_text(item, "paradigm", 32, "MI").upper()
        supported = {part.strip() for part in paradigm.split("/") if part.strip()}
        if not supported or not supported <= _PARADIGMS:
            raise PluginError(f"pipeline {pipeline_id} 的 paradigm 非法: {paradigm}")
        graph = item.get("graph")
        if not isinstance(graph, dict):
            raise PluginError(f"pipeline {pipeline_id} 缺少 graph")
        pipelines.append(
            {
                "id": pipeline_id,
                "name": pipeline_name,
                "paradigm": "/".join(sorted(supported)),
                "description": _require_text(item, "description", 500),
                "graph": graph,
            }
        )
    tags = [str(tag).strip()[:24] for tag in raw.get("tags") or [] if str(tag).strip()]
    if len(tags) > 8:
        raise PluginError("tags 最多 8 个")
    manifest = {
        "id": identifier,
        "name": name,
        "version": version,
        "description": _require_text(raw, "description", 1000),
        "author": _require_text(raw, "author", 80, "未署名"),
        "homepage": _require_text(raw, "homepage", 300),
        "license": _require_text(raw, "license", 80, "未声明"),
        "runtime": runtime,
        "permissions": permissions,
        "entry": entry,
        "tags": tags,
        "pipelines": pipelines,
    }
    # Only emit schema fields that the package declared so that legacy v1
    # manifests keep their original digest and remain verifiable.
    if "schema_version" in raw:
        manifest["schema_version"] = schema_version
    if schema_version >= 2:
        manifest["engine_min_version"] = engine_min_version
        manifest["api_version"] = api_version
    return manifest


@dataclass
class _ActivePlugin:
    manifest: dict[str, Any]
    directory: Path
    definitions: dict[str, dict[str, Any]]
    node_types: list[str]
    module_package: str = ""


class PluginAPI:
    """Registration surface passed to ``register(api)`` in Python plugins."""

    def __init__(self, plugin_id: str, directory: Path):
        self.plugin_id = plugin_id
        self.directory = directory
        self.nodes: list[dict[str, Any]] = []

    def register_node(self, spec: dict[str, Any]) -> None:
        if not isinstance(spec, dict):
            raise PluginError("register_node 需要一个对象")
        node_type = str(spec.get("type") or "").strip()
        if not re.match(rf"^plugin\.{re.escape(self.plugin_id)}\.[a-z0-9_]{{2,40}}$", node_type):
            raise PluginError(f"插件节点类型必须以 plugin.{self.plugin_id}. 开头: {node_type!r}")
        if len(self.nodes) >= 16:
            raise PluginError("单个插件最多注册 16 个节点")
        if any(item.get("type") == node_type for item in self.nodes):
            raise PluginError(f"插件节点类型重复: {node_type}")
        self.nodes.append(dict(spec))


class PluginManager:
    """Install, verify, enable and expose local plugins."""

    def __init__(
        self,
        root: str | Path,
        catalog_dir: str | Path | None = None,
        trusted: bool | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.catalog_dir = Path(
            catalog_dir or Path(__file__).resolve().parent / "plugin_catalog"
        ).expanduser().resolve()
        self.trusted = (
            _env_trusted() if trusted is None else bool(trusted)
        )
        self._lock = threading.RLock()
        self._active: dict[str, _ActivePlugin] = {}
        self._errors: dict[str, str] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._load_enabled()

    @property
    def mode(self) -> str:
        return "trusted" if self.trusted else "restricted"

    def _env_trusted(self) -> bool:
        return _env_trusted()

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _read_index(self) -> dict[str, Any]:
        path = self._index_path()
        if not path.exists():
            return {"version": 1, "plugins": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "plugins": {}}
        plugins = payload.get("plugins")
        return {
            "version": 1,
            "plugins": plugins if isinstance(plugins, dict) else {},
        }

    def _write_index(self, payload: dict[str, Any]) -> None:
        temp = self.root / f".index-{uuid.uuid4().hex}.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, self._index_path())

    def _set_state(self, plugin_id: str, **values: Any) -> None:
        index = self._read_index()
        entry = index["plugins"].get(plugin_id) or {}
        entry.update(values)
        index["plugins"][plugin_id] = entry
        self._write_index(index)

    def _load_enabled(self) -> None:
        index = self._read_index()
        for plugin_id, entry in index["plugins"].items():
            if not entry.get("enabled"):
                continue
            try:
                manifest, directory = self._read_installed(plugin_id)
                self._active[plugin_id] = self._activate(manifest, directory)
                self._errors.pop(plugin_id, None)
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not block startup
                self._errors[plugin_id] = str(exc)
                self._set_state(plugin_id, enabled=False, last_error=str(exc))

    def _read_installed(self, plugin_id: str) -> tuple[dict[str, Any], Path]:
        if not PLUGIN_ID.match(plugin_id or ""):
            raise PluginError(f"插件 ID 非法: {plugin_id!r}")
        directory = (self.root / plugin_id).resolve()
        if directory.parent != self.root:
            raise PluginError(f"插件目录越界: {plugin_id}")
        manifest_path = directory / "plugin.json"
        if not manifest_path.is_file():
            raise PluginError(f"插件未安装: {plugin_id}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PluginError(f"插件清单无法读取: {plugin_id}") from exc
        manifest = normalize_manifest(raw, plugin_id=plugin_id)
        manifest.update(
            {
                key: raw[key]
                for key in (*_COMPUTED_FIELDS, "files", "files_verified")
                if key in raw
            }
        )
        return manifest, directory

    def _installed_dir(self, plugin_id: str) -> Path:
        directory = (self.root / plugin_id).resolve()
        if directory.parent != self.root:
            raise PluginError("插件目录越界")
        return directory

    def _verify_package(self, manifest: dict[str, Any], directory: Path) -> None:
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise PluginError("插件清单缺少文件哈希，无法验证")
        if _manifest_digest(manifest) != manifest.get("manifest_sha256"):
            raise PluginError("插件清单哈希不匹配，可能被篡改")
        if _package_digest(files) != manifest.get("package_sha256"):
            raise PluginError("插件包哈希不匹配")
        expected = {str(name) for name in files}
        actual = set()
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(directory).as_posix()
            if relative == "plugin.json":
                continue
            if path.is_symlink():
                raise PluginError(f"插件不允许符号链接: {relative}")
            actual.add(relative)
        if actual != expected:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            raise PluginError(
                "插件文件集合与清单不一致"
                + (f"；多出 {extra[:3]}" if extra else "")
                + (f"；缺少 {missing[:3]}" if missing else "")
            )
        for relative, digest in files.items():
            payload = (directory / relative).read_bytes()
            if _sha256(payload) != digest:
                raise PluginError(f"插件文件哈希不匹配: {relative}")

    def _load_python(self, manifest: dict[str, Any], directory: Path) -> tuple[str, list[dict[str, Any]]]:
        package_name = f"_neurokiln_plugin_{manifest['id'].replace('-', '_')}"
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        module_name = f"{package_name}.entry"
        entry_path = (directory / manifest["entry"]).resolve()
        if not entry_path.is_file() or directory.resolve() not in entry_path.parents:
            raise PluginError("Python 入口文件不存在或越界")
        spec = spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            raise PluginError("无法加载 Python 入口")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if not callable(register):
                raise PluginError("Python 插件必须提供 register(api)")
            api = PluginAPI(manifest["id"], directory)
            register(api)
        except Exception:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous_bytecode
        return package_name, api.nodes

    def _activate(self, manifest: dict[str, Any], directory: Path) -> _ActivePlugin:
        _ensure_compatible(manifest)
        self._verify_package(manifest, directory)
        plugin_id = manifest["id"]
        if manifest["runtime"] == "python" and not self.trusted:
            raise PluginError(
                f"插件 {plugin_id} 包含 Python 代码，受限模式拒绝加载；"
                "请由管理员审阅后以 NEUROKILN_PLUGIN_TRUST=1 启动"
            )
        node_types: list[str] = []
        package_name = ""
        try:
            if manifest["runtime"] == "python":
                package_name, node_specs = self._load_python(manifest, directory)
                node_types = register_plugin_node_specs(node_specs)
            definitions: dict[str, dict[str, Any]] = {}
            for pipeline in manifest["pipelines"]:
                identifier = f"plugin:{plugin_id}/{pipeline['id']}"
                normalized = normalize_graph(pipeline["graph"])
                build_graph_pipeline(identifier, normalized)
                definitions[identifier] = {
                    "id": identifier,
                    "name": pipeline["name"],
                    "family": "plugin",
                    "paradigm": pipeline["paradigm"],
                    "note": pipeline.get("description") or manifest["description"],
                    "graph": normalized,
                    "plugin": {
                        "id": plugin_id,
                        "version": manifest["version"],
                        "schema_version": int(manifest.get("schema_version", 1)),
                        "api_version": int(manifest.get("api_version", 1)),
                        "package_sha256": manifest["package_sha256"],
                        "manifest_sha256": manifest.get("manifest_sha256", ""),
                    },
                }
        except Exception:
            if node_types:
                unregister_plugin_node_specs(f"plugin.{plugin_id}.")
            if package_name:
                for name in list(sys.modules):
                    if name == package_name or name.startswith(package_name + "."):
                        sys.modules.pop(name, None)
            raise
        return _ActivePlugin(
            manifest=manifest,
            directory=directory,
            definitions=definitions,
            node_types=node_types,
            module_package=package_name,
        )

    def _deactivate(self, plugin_id: str) -> None:
        active = self._active.pop(plugin_id, None)
        if active is None:
            return
        unregister_plugin_node_specs(f"plugin.{plugin_id}.")
        if active.module_package:
            for name in list(sys.modules):
                if name == active.module_package or name.startswith(active.module_package + "."):
                    sys.modules.pop(name, None)

    def _public_item(self, manifest: dict[str, Any], *, error: str | None = None) -> dict[str, Any]:
        plugin_id = manifest["id"]
        active = self._active.get(plugin_id)
        index = self._read_index()["plugins"].get(plugin_id) or {}
        return {
            "id": plugin_id,
            "name": manifest["name"],
            "version": manifest["version"],
            "description": manifest.get("description", ""),
            "author": manifest.get("author", ""),
            "homepage": manifest.get("homepage", ""),
            "license": manifest.get("license", ""),
            "runtime": manifest["runtime"],
            "permissions": list(manifest.get("permissions") or []),
            "tags": list(manifest.get("tags") or []),
            "pipelines": [
                {
                    "id": f"plugin:{plugin_id}/{item['id']}",
                    "name": item["name"],
                    "paradigm": item["paradigm"],
                    "description": item.get("description", ""),
                }
                for item in manifest["pipelines"]
            ],
            "enabled": bool(index.get("enabled")) and active is not None,
            "active": active is not None,
            "verified": active is not None,
            "requires_trust": manifest["runtime"] == "python",
            "package_sha256": manifest.get("package_sha256") or index.get("package_sha256") or "",
            "manifest_sha256": manifest.get("manifest_sha256") or index.get("manifest_sha256") or "",
            "files": sorted((manifest.get("files") or {}).keys()),
            "mode": self.mode,
            "installed_at": index.get("installed_at"),
            "source": index.get("source", "upload"),
            **_compat_fields(manifest),
            "error": error if error is not None else (self._errors.get(plugin_id) or index.get("last_error")),
        }

    def list_installed(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for directory in sorted(self.root.iterdir()):
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                manifest_path = directory / "plugin.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifest, _ = self._read_installed(directory.name)
                except (PluginError, OSError) as exc:
                    items.append(
                        {
                            "id": directory.name,
                            "name": directory.name,
                            "version": "—",
                            "description": "",
                            "author": "",
                            "runtime": "unknown",
                            "permissions": [],
                            "tags": [],
                            "pipelines": [],
                            "enabled": False,
                            "active": False,
                            "verified": False,
                            "requires_trust": False,
                            "installed_at": None,
                            "source": "unknown",
                            "schema_version": 0,
                            "engine_min_version": "",
                            "api_version": 0,
                            "compatible": False,
                            "compatibility": "清单无法读取",
                            "error": str(exc),
                        }
                    )
                    continue
                items.append(self._public_item(manifest))
            return items

    def catalog(self) -> list[dict[str, Any]]:
        if not self.catalog_dir.is_dir():
            return []
        installed = {item["id"]: item for item in self.list_installed()}
        items = []
        for directory in sorted(self.catalog_dir.iterdir()):
            manifest_path = directory / "plugin.json"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = normalize_manifest(
                    json.loads(manifest_path.read_text(encoding="utf-8")),
                    plugin_id=directory.name,
                )
            except (PluginError, OSError, json.JSONDecodeError):
                continue
            current = installed.get(manifest["id"])
            items.append(
                {
                    "id": manifest["id"],
                    "name": manifest["name"],
                    "version": manifest["version"],
                    "description": manifest["description"],
                    "author": manifest["author"],
                    "license": manifest["license"],
                    "runtime": manifest["runtime"],
                    "permissions": manifest["permissions"],
                    "tags": manifest["tags"],
                    "pipelines": [
                        {
                            "id": f"plugin:{manifest['id']}/{item['id']}",
                            "name": item["name"],
                            "paradigm": item["paradigm"],
                            "description": item.get("description", ""),
                        }
                        for item in manifest["pipelines"]
                    ],
                    "installed": current is not None,
                    "installed_version": current["version"] if current else None,
                    "source": "bundled",
                    **_compat_fields(manifest),
                }
            )
        return items

    def status(self) -> dict[str, Any]:
        plugins = self.list_installed()
        return {
            "mode": self.mode,
            "trusted": self.trusted,
            "engine_version": ENGINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "supported_schema_versions": sorted(SUPPORTED_SCHEMA_VERSIONS),
            "node_api_version": NODE_API_VERSION,
            "root": str(self.root),
            "catalog_dir": str(self.catalog_dir),
            "supported_runtimes": ["graph", "python"] if self.trusted else ["graph"],
            "allowed_permissions": ["graph", "nodes"] if self.trusted else ["graph"],
            "plugins": plugins,
            "catalog": self.catalog(),
            "counts": {
                "installed": len(plugins),
                "enabled": sum(1 for item in plugins if item["enabled"]),
                "errors": sum(1 for item in plugins if item.get("error")),
            },
        }

    def definitions(self) -> list[dict[str, Any]]:
        with self._lock:
            definitions = []
            for active in self._active.values():
                definitions.extend(
                    json.loads(json.dumps(definition))
                    for definition in active.definitions.values()
                )
            return sorted(definitions, key=lambda item: item["id"])

    def definition(self, identifier: str) -> dict[str, Any]:
        if not identifier.startswith("plugin:"):
            raise PluginError(f"不是插件算法: {identifier}")
        with self._lock:
            for active in self._active.values():
                definition = active.definitions.get(identifier)
                if definition is not None:
                    return json.loads(json.dumps(definition))
        raise PluginError(f"插件算法不可用（插件未安装或未启用）: {identifier}")

    def require_definition(self, definition: dict[str, Any]) -> None:
        meta = definition.get("plugin") or {}
        plugin_id = str(meta.get("id") or "")
        if not plugin_id:
            return
        with self._lock:
            active = self._active.get(plugin_id)
        if active is None:
            raise PluginError(
                f"复现该实验需要启用插件 {plugin_id}；请先在插件市场安装并启用"
            )
        if str(meta.get("version") or "") and active.manifest["version"] != str(meta["version"]):
            raise PluginError(
                f"插件 {plugin_id} 版本不匹配：需要 {meta['version']}，"
                f"当前 {active.manifest['version']}"
            )
        if (
            str(meta.get("package_sha256") or "")
            and active.manifest.get("package_sha256") != str(meta["package_sha256"])
        ):
            raise PluginError(f"插件 {plugin_id} 包哈希与实验记录不一致，拒绝复现")
        if (
            str(meta.get("manifest_sha256") or "")
            and active.manifest.get("manifest_sha256") != str(meta["manifest_sha256"])
        ):
            raise PluginError(f"插件 {plugin_id} 清单哈希与实验记录不一致，拒绝复现")

    def enable(self, plugin_id: str) -> dict[str, Any]:
        with self._lock:
            manifest, directory = self._read_installed(plugin_id)
            if plugin_id in self._active:
                return self._public_item(manifest)
            if manifest["runtime"] == "python" and not self.trusted:
                raise PluginError("受限模式不能启用 Python 插件")
            try:
                active = self._activate(manifest, directory)
            except Exception as exc:
                self._errors[plugin_id] = str(exc)
                self._set_state(plugin_id, enabled=False, last_error=str(exc))
                raise PluginError(f"插件启用失败: {exc}") from exc
            self._active[plugin_id] = active
            self._errors.pop(plugin_id, None)
            self._set_state(plugin_id, enabled=True, last_error=None)
            return self._public_item(manifest)

    def disable(self, plugin_id: str) -> dict[str, Any]:
        with self._lock:
            manifest, _ = self._read_installed(plugin_id)
            self._deactivate(plugin_id)
            self._errors.pop(plugin_id, None)
            self._set_state(plugin_id, enabled=False, last_error=None)
            return self._public_item(manifest)

    def _install_files(
        self,
        files: dict[str, bytes],
        *,
        source: str,
        enable: bool,
    ) -> dict[str, Any]:
        if not files:
            raise PluginError("插件包为空")
        if len(files) > _MAX_FILES:
            raise PluginError(f"插件文件数不能超过 {_MAX_FILES}")
        total = sum(len(payload) for payload in files.values())
        if total > _MAX_TOTAL_BYTES:
            raise PluginError("插件包解压后超过 60 MiB")
        files = _strip_single_root(files)
        if "plugin.json" not in files:
            raise PluginError("插件包根目录缺少 plugin.json")
        try:
            raw = json.loads(files["plugin.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginError("plugin.json 不是合法的 UTF-8 JSON") from exc
        manifest = normalize_manifest(raw)
        _ensure_compatible(manifest)
        if manifest["runtime"] == "python" and not self.trusted:
            raise PluginError(
                "受限模式仅允许安装声明式图插件；"
                "如确需 Python 插件，请管理员审阅后以 NEUROKILN_PLUGIN_TRUST=1 启动"
            )
        plugin_id = manifest["id"]
        with self._lock:
            final = self._installed_dir(plugin_id)
            if final.exists():
                raise PluginError(f"插件 {plugin_id} 已安装，请先卸载或升级")
            stage = self.root / f".tmp-{uuid.uuid4().hex}"
            stage.mkdir(parents=True, exist_ok=False)
            try:
                hashes = {}
                for name, payload in files.items():
                    relative = _safe_relative(name)
                    if relative == "plugin.json":
                        continue
                    if len(payload) > _MAX_FILE_BYTES:
                        raise PluginError(f"文件 {relative} 超过 20 MiB")
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    hashes[relative] = _sha256(payload)
                manifest["files"] = dict(sorted(hashes.items()))
                manifest["package_sha256"] = _package_digest(manifest["files"])
                manifest["manifest_sha256"] = _manifest_digest(manifest)
                manifest["installed_at"] = _timestamp()
                manifest["source"] = source
                (stage / "plugin.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(stage, final)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            self._set_state(
                plugin_id,
                enabled=False,
                installed_at=manifest["installed_at"],
                source=source,
                last_error=None,
            )
            if enable:
                try:
                    self.enable(plugin_id)
                except PluginError as exc:
                    self._errors[plugin_id] = str(exc)
                    self._set_state(plugin_id, enabled=False, last_error=str(exc))
            manifest, _ = self._read_installed(plugin_id)
            return self._public_item(manifest)

    def install_bundled(self, plugin_id: str, *, enable: bool = True) -> dict[str, Any]:
        if not PLUGIN_ID.match(plugin_id or ""):
            raise PluginError("插件 ID 非法")
        directory = (self.catalog_dir / plugin_id).resolve()
        if directory.parent != self.catalog_dir or not directory.is_dir():
            raise PluginError(f"本地市场没有插件: {plugin_id}")
        files = _read_directory(directory)
        return self._install_files(files, source="bundled", enable=enable)

    def install_zip(
        self, payload: bytes, *, filename: str = "plugin.zip", enable: bool = True
    ) -> dict[str, Any]:
        files = _read_zip_files(payload, filename)
        return self._install_files(files, source="upload", enable=enable)

    def uninstall(self, plugin_id: str) -> None:
        with self._lock:
            manifest, directory = self._read_installed(plugin_id)
            self._deactivate(plugin_id)
            shutil.rmtree(directory)
            index = self._read_index()
            index["plugins"].pop(plugin_id, None)
            self._write_index(index)
            self._errors.pop(plugin_id, None)
            _ = manifest


def validate_plugin_source(
    source: str | Path, *, engine_version: str = ENGINE_VERSION
) -> dict[str, Any]:
    """Conformance-check a plugin directory, ZIP archive or manifest file.

    Never installs or executes plugin code; Python-plugin DAG checks that need
    registered plugin nodes are reported as deferred instead.
    """

    path = Path(source).expanduser()
    if not path.exists():
        raise PluginError(f"路径不存在: {path}")
    errors: list[str] = []
    warnings: list[str] = []
    pipelines: list[dict[str, Any]] = []
    package: dict[str, Any] = {
        "declared": False,
        "verified": False,
        "files": 0,
        "missing": [],
        "extra": [],
        "mismatched": [],
        "manifest_sha256": "",
    }
    files: dict[str, bytes] | None = None
    if path.is_dir():
        kind = "directory"
        try:
            files = _read_directory(path)
        except PluginError as exc:
            errors.append(str(exc))
    elif zipfile.is_zipfile(path):
        kind = "zip"
        try:
            files = _strip_single_root(
                _read_zip_files(path.read_bytes(), path.name)
            )
        except PluginError as exc:
            errors.append(str(exc))
    elif path.suffix.lower() == ".json":
        kind = "manifest"
    else:
        raise PluginError("请提供插件目录、ZIP 包或 plugin.json 文件")

    raw: dict[str, Any] | None = None
    if files is not None:
        payload = files.get("plugin.json")
        if payload is None:
            errors.append("插件包根目录缺少 plugin.json")
        else:
            try:
                raw = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors.append("plugin.json 不是合法的 UTF-8 JSON")
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append(f"无法读取清单: {exc}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append("plugin.json 不是合法的 UTF-8 JSON")

    manifest: dict[str, Any] | None = None
    if raw is not None:
        try:
            manifest = normalize_manifest(raw)
        except PluginError as exc:
            errors.append(str(exc))

    compatible = False
    compatibility = ""
    manifest_view: dict[str, Any] | None = None
    if manifest is not None:
        compatible, compatibility = plugin_compatibility(manifest, engine_version)
        if not compatible:
            errors.append(f"与当前引擎不兼容: {compatibility}")
        manifest_view = {
            key: manifest[key]
            for key in (
                "id", "name", "version", "description", "author", "license",
                "runtime", "permissions", "tags", "entry",
            )
        }
        manifest_view.update(
            {
                "schema_version": int(manifest.get("schema_version", 1)),
                "engine_min_version": manifest.get("engine_min_version", ""),
                "api_version": int(manifest.get("api_version", 1)),
            }
        )
        manifest_view["pipelines"] = [
            {
                "id": f"plugin:{manifest['id']}/{item['id']}",
                "name": item["name"],
                "paradigm": item["paradigm"],
            }
            for item in manifest["pipelines"]
        ]
        if manifest["runtime"] == "python":
            warnings.append(
                "Python 插件将在进程内执行且没有沙箱，只能在可信模式下启用，请先审阅源码"
            )

    if manifest is not None:
        declared = raw.get("files") if isinstance(raw, dict) else None
        actual = {
            name: _sha256(payload)
            for name, payload in (files or {}).items()
            if name != "plugin.json"
        }
        if isinstance(declared, dict):
            try:
                expected = {
                    _safe_relative(str(name)): str(digest)
                    for name, digest in declared.items()
                }
            except PluginError as exc:
                errors.append(str(exc))
                expected = {}
            package["declared"] = True
            package["files"] = len(expected)
            package["missing"] = sorted(set(expected) - set(actual))
            package["extra"] = sorted(set(actual) - set(expected))
            package["mismatched"] = sorted(
                name
                for name in set(expected) & set(actual)
                if expected[name] != actual[name]
            )
            if files is None:
                warnings.append("清单声明了文件哈希，但校验源不是完整插件包")
            else:
                if package["missing"]:
                    errors.append(f"插件包缺少文件: {package['missing'][:5]}")
                if package["extra"]:
                    errors.append(f"插件包存在未声明文件: {package['extra'][:5]}")
                if package["mismatched"]:
                    errors.append(f"文件哈希不匹配: {package['mismatched'][:5]}")
                package["verified"] = (
                    not package["missing"]
                    and not package["extra"]
                    and not package["mismatched"]
                )
            stored_manifest = raw.get("manifest_sha256")
            if stored_manifest:
                if _manifest_digest(raw) == stored_manifest:
                    package["manifest_sha256"] = "verified"
                else:
                    package["manifest_sha256"] = "mismatch"
                    errors.append("清单哈希不匹配，可能被篡改")
        else:
            warnings.append("清单未封存文件哈希（files）；安装时会重新生成")

        if manifest["runtime"] == "python" and files is not None:
            if manifest["entry"] not in files:
                errors.append(f"缺少 Python 入口文件: {manifest['entry']}")

        plugin_node_prefix = f"plugin.{manifest['id']}."
        for pipeline in manifest["pipelines"]:
            item: dict[str, Any] = {
                "id": f"plugin:{manifest['id']}/{pipeline['id']}",
                "name": pipeline["name"],
                "paradigm": pipeline["paradigm"],
                "status": "ok",
                "message": "",
            }
            node_types = [
                str(node.get("type") or "")
                for node in pipeline["graph"].get("nodes") or []
                if isinstance(node, dict)
            ]
            deferred = manifest["runtime"] == "python" and any(
                node_type.startswith(plugin_node_prefix) for node_type in node_types
            )
            if deferred:
                item["status"] = "deferred"
                item["message"] = "包含插件注册节点，需在可信模式加载后校验"
                warnings.append(f"{item['id']}: {item['message']}")
            else:
                try:
                    build_graph_pipeline(
                        item["id"], normalize_graph(pipeline["graph"])
                    )
                except (GraphValidationError, ValueError) as exc:
                    item["status"] = "error"
                    item["message"] = str(exc)
                    errors.append(f"{item['id']}: {exc}")
            pipelines.append(item)

    return {
        "valid": not errors,
        "source": str(path.resolve()),
        "kind": kind,
        "engine_version": engine_version,
        "schema_version": SCHEMA_VERSION,
        "supported_schema_versions": sorted(SUPPORTED_SCHEMA_VERSIONS),
        "node_api_version": NODE_API_VERSION,
        "manifest": manifest_view,
        "compatible": compatible,
        "compatibility": compatibility,
        "package": package,
        "pipelines": pipelines,
        "errors": errors,
        "warnings": warnings,
    }


def _env_trusted() -> bool:
    return os.environ.get("NEUROKILN_PLUGIN_TRUST", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_directory(directory: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix()
        _safe_relative(relative)
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise PluginError(f"文件 {relative} 超过 20 MiB")
        total += path.stat().st_size
        if total > _MAX_TOTAL_BYTES:
            raise PluginError("插件目录超过 60 MiB")
        files[relative] = path.read_bytes()
    return files


__all__ = [
    "PluginAPI",
    "PluginError",
    "PluginManager",
    "normalize_manifest",
    "plugin_compatibility",
    "validate_plugin_source",
]
