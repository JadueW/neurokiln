# syntax=docker/dockerfile:1

FROM node:22-alpine AS web-build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY engine /app/engine
COPY server /app/server

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/server/requirements.txt \
    && pip install --no-deps /app/engine

COPY --from=web-build /app/web/dist /app/web/dist

RUN useradd --create-home --uid 10001 neurokiln \
    && mkdir -p /data \
    && chown -R neurokiln:neurokiln /app /data

USER neurokiln
ENV HOME=/home/neurokiln \
    NEUROKILN_DATA_DIR=/data \
    NEUROKILN_MNE_DATA=/data/mne \
    NEUROKILN_WEB_DIST=/app/web/dist \
    MNE_DATA=/data/mne

VOLUME ["/data"]
EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/server", "--host", "0.0.0.0", "--port", "8100"]
