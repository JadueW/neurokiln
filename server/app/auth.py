from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_ITERATIONS = 310_000

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TEXT NOT NULL,
    last_login_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, salt_hex: str, expected_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return hmac.compare_digest(digest.hex(), expected_hex)


def validate_credentials(username: str, password: str) -> None:
    if not _USERNAME.match(username or ""):
        raise ValueError("用户名需为 3–32 位字母、数字、下划线、点或短横线")
    if len(password or "") < 8:
        raise ValueError("密码至少需要 8 位")


class AuthService:
    """Local account store with signed, expiring bearer tokens.

    Only the standard library is used: PBKDF2-HMAC-SHA256 for passwords and
    HMAC-SHA256 over a compact JSON payload for tokens. The signing secret is
    generated on first use and never committed.
    """

    def __init__(self, db_path: str | Path, secret_path: str | Path, ttl_seconds: int = 12 * 3600):
        self.path = Path(db_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path = Path(secret_path).expanduser().resolve()
        self.ttl_seconds = int(ttl_seconds)
        self._secret = self._load_secret()
        with self._connect() as conn:
            conn.executescript(AUTH_SCHEMA)

    def _load_secret(self) -> bytes:
        environment = os.environ.get("NEUROKILN_SECRET")
        if environment:
            return environment.encode("utf-8")
        if self.secret_path.exists():
            value = self.secret_path.read_bytes().strip()
            if len(value) >= 32:
                return value
        secret = secrets.token_bytes(32)
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path.write_bytes(secret.hex().encode("ascii"))
        return secret

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def setup_required(self) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0

    def user_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, role: str = "member") -> dict[str, Any]:
        validate_credentials(username, password)
        role = role if role in {"admin", "member"} else "member"
        password_hash, salt = hash_password(password)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users(username, password_hash, salt, role, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (username, password_hash, salt, role, _now()),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已存在") from exc
        return {"id": user_id, "username": username, "role": role}

    def bootstrap(self, username: str, password: str) -> dict[str, Any]:
        validate_credentials(username, password)
        password_hash, salt = hash_password(password)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count:
                raise ValueError("系统已经完成初始化，请直接登录")
            cursor = conn.execute(
                """
                INSERT INTO users(username, password_hash, salt, role, created_at)
                VALUES(?, ?, ?, 'admin', ?)
                """,
                (username, password_hash, salt, _now()),
            )
            user_id = int(cursor.lastrowid)
        return {"id": user_id, "username": username, "role": "admin"}

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None or not verify_password(password, row["salt"], row["password_hash"]):
                return None
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), row["id"])
            )
            return {"id": row["id"], "username": row["username"], "role": row["role"]}

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return _public_user(row) if row is not None else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [_public_user(row) for row in rows]

    def delete_user(self, user_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                return False
            if row["role"] == "admin":
                admins = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin'"
                ).fetchone()[0]
                if admins <= 1:
                    raise ValueError("不能删除最后一个管理员")
            conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
        return True

    def issue_token(self, user: dict[str, Any]) -> dict[str, Any]:
        issued = int(time.time())
        expires = issued + self.ttl_seconds
        payload = {
            "sub": int(user["id"]),
            "usr": user["username"],
            "role": user["role"],
            "iat": issued,
            "exp": expires,
        }
        body = _b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        signature = _b64encode(
            hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        )
        return {
            "token": f"{body}.{signature}",
            "token_type": "bearer",
            "expires_at": expires,
            "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
        }

    def verify_token(self, token: str) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        body, signature = token.rsplit(".", 1)
        try:
            expected = _b64encode(
                hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
            )
        except (UnicodeEncodeError, ValueError):
            return None
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = json.loads(_b64decode(body))
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            expires = int(payload.get("exp", 0))
            payload["sub"] = int(payload["sub"])
        except (TypeError, ValueError, KeyError):
            return None
        if expires <= int(time.time()):
            return None
        return payload


def _public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


__all__ = ["AuthService", "hash_password", "validate_credentials", "verify_password"]
