"""Structured API errors.

Every error response carries a stable machine-readable ``code`` next to the
human-readable ``detail`` so the web UI can translate messages without
matching Chinese strings. Codes are part of the public API surface: add new
ones instead of renaming existing codes.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("neurokiln.api")

_STATUS_CODES = {
    400: "request.invalid",
    401: "auth.required",
    403: "auth.admin_required",
    404: "request.not_found",
    409: "request.conflict",
    413: "request.too_large",
    422: "request.validation",
    429: "request.rate_limited",
    500: "server.error",
    502: "service.upstream_failed",
    503: "service.unavailable",
}


class ApiError(StarletteHTTPException):
    def __init__(self, status_code: int, detail: str, code: str):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


def _default_code(status_code: int) -> str:
    if status_code in _STATUS_CODES:
        return _STATUS_CODES[status_code]
    return "server.error" if status_code >= 500 else "request.failed"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "code": "request.validation"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": _default_code(exc.status_code)},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "code": "server.error"},
        )
