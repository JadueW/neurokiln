import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = tempfile.mkdtemp(prefix="neurokiln-xai-api-")
os.environ["NEUROKILN_DATA_DIR"] = DATA_DIR
sys.path.insert(0, str(ROOT / "server"))

import pytest
from fastapi.testclient import TestClient

from app.main import app

PAYLOAD = {
    "dataset_id": "synthetic",
    "subject_ids": ["S01"],
    "protocol": "within_subject",
    "folds": 3,
    "repeats": 1,
    "seeds": [0],
    "pipeline": "csp_lda",
    "fold": 0,
    "repeat": 0,
    "window_ms": 250,
    "max_blocks": 24,
    "gradient_method": "none",
    "include_audit": True,
    "include_sanity": False,
    "options": {"n_trials_per_class": 30},
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def headers(client):
    response = client.post(
        "/api/auth/bootstrap",
        json={"username": "xaiadmin", "password": "xai-password-123"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def wait_job(client, headers, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def teardown_module(module):
    shutil.rmtree(DATA_DIR, ignore_errors=True)


def test_explanation_job_lifecycle(client, headers):
    assert client.post("/api/explanations/jobs", json=PAYLOAD).status_code == 401

    submitted = client.post("/api/explanations/jobs", json=PAYLOAD, headers=headers)
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, headers, submitted.json()["id"])
    assert job["status"] == "completed", job.get("error")

    explanation_id = job["result"]["explanation_id"]
    assert explanation_id > 0

    listing = client.get("/api/explanations", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["id"] == explanation_id

    detail = client.get(f"/api/explanations/{explanation_id}", headers=headers)
    assert detail.status_code == 200
    document = detail.json()["document"]
    assert document["format"] == "neurokiln.explain"
    assert document["provenance"]["background"] == "train_fold_only"
    assert document["faithfulness"]["score_unit"] == "log_odds"
    assert document["occlusion"]["score_unit"] == "log_odds"
    images = detail.json()["images"]
    assert images and all(
        payload.startswith("data:image/png;base64,") for payload in images.values()
    )

    asset = document["occlusion"]["topomap"]
    download = client.get(
        f"/api/explanations/{explanation_id}/assets/{asset}", headers=headers
    )
    assert download.status_code == 200
    assert download.content[:8] == b"\x89PNG\r\n\x1a\n"

    traversal = client.get(
        f"/api/explanations/{explanation_id}/assets/../../test.db", headers=headers
    )
    assert traversal.status_code in {404, 422}

    assert client.delete(
        f"/api/explanations/{explanation_id}"
    ).status_code == 401
    removed = client.delete(f"/api/explanations/{explanation_id}", headers=headers)
    assert removed.status_code == 200
    assert client.get(
        f"/api/explanations/{explanation_id}", headers=headers
    ).status_code == 404

    empty = client.get(
        "/api/explanations/aggregate", params={"pipeline": "csp_lda"}, headers=headers
    )
    assert empty.status_code == 400
    assert "没有可聚合" in empty.json()["detail"]
    assert empty.json()["code"] == "explain.aggregate_empty"


def test_api_errors_carry_stable_codes(client, headers):
    unauthorized = client.get("/api/explanations")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["code"] == "auth.token_invalid"

    wrong = client.post(
        "/api/auth/login",
        json={"username": "xaiadmin", "password": "wrong-password-999"},
    )
    assert wrong.status_code == 401
    assert wrong.json()["code"] == "auth.invalid_credentials"

    missing = client.get("/api/explanations/999999", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["code"] == "explain.not_found"

    invalid = client.get(
        "/api/explanations/aggregate", params={"limit": 1}, headers=headers
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "request.validation"

    unknown = client.get("/api/no-such-endpoint", headers=headers)
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "request.not_found"


def test_cross_fold_explanation_aggregate(client, headers):
    for fold in (0, 1):
        submitted = client.post(
            "/api/explanations/jobs",
            json={**PAYLOAD, "fold": fold},
            headers=headers,
        )
        assert submitted.status_code == 200, submitted.text
        job = wait_job(client, headers, submitted.json()["id"])
        assert job["status"] == "completed", job.get("error")

    assert client.get("/api/explanations/aggregate").status_code == 401

    response = client.get(
        "/api/explanations/aggregate",
        params={"pipeline": "csp_lda", "dataset_id": "synthetic"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    aggregate = response.json()
    assert aggregate["format"] == "neurokiln.explain.aggregate"
    assert aggregate["pipeline"]["id"] == "csp_lda"
    assert aggregate["provenance"]["n_explanations"] == 2
    assert aggregate["provenance"]["n_unique_splits"] == 2
    assert sorted(row["fold"] for row in aggregate["splits"]) == [0, 1]
    assert aggregate["occlusion"]["available_count"] == 2
    channel_n = aggregate["occlusion"]["channel_importance"]["n"]
    assert channel_n and all(value == 2 for value in channel_n)
    assert aggregate["faithfulness"]["deletion_auc"]["n"] == 2
    assert aggregate["top_channels"]
    assert any(entry["id"] == "csp_patterns" for entry in aggregate["intrinsic"])

    mismatch = client.get(
        "/api/explanations/aggregate",
        params={"pipeline": "csp_lda", "dataset_id": "other"},
        headers=headers,
    )
    assert mismatch.status_code == 400

