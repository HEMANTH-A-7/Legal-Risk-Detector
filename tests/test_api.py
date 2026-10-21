import io

import pytest

import app as app_module

CONTRACT = (
    "The contractor shall indemnify all parties. "
    "Late payments are subject to a 1.5% monthly surcharge. "
    "This page intentionally lists the parties."
)


@pytest.fixture
def client(monkeypatch):
    # Keyword tier only: exercises the full request path without model weights.
    monkeypatch.setenv("RISK_DETECTOR", "keyword")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_analyze_text(client):
    res = client.post("/analyze", data={"text": CONTRACT})
    assert res.status_code == 200
    body = res.get_json()
    assert body["summary"]["total_sentences"] == 3
    assert body["summary"]["risky_count"] == 2
    assert body["ccic"]["mode"] == "keyword"
    assert body["ccic"]["detector_distribution"]["keyword"] == 2
    assert sum(body["summary"]["severity_distribution"].values()) == 2


def test_analyze_txt_upload(client):
    data = {"file": (io.BytesIO(CONTRACT.encode()), "contract.txt")}
    res = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["summary"]["risky_count"] == 2


def test_rejects_unsupported_file_type(client):
    data = {"file": (io.BytesIO(b"x"), "contract.exe")}
    res = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_rejects_empty_request(client):
    assert client.post("/analyze", data={"text": "   "}).status_code == 400
    assert client.post("/analyze", data={}).status_code == 400


def test_rejects_oversized_upload(client):
    app_module.app.config["MAX_CONTENT_LENGTH"] = 1024
    try:
        data = {"file": (io.BytesIO(b"a" * 4096), "big.txt")}
        res = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert res.status_code == 413
    finally:
        app_module.app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def test_internal_errors_do_not_leak_tracebacks(client, monkeypatch):
    def boom(sentences):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module, "analyze_risks", boom)
    res = client.post("/analyze", data={"text": CONTRACT})
    assert res.status_code == 500
    body = res.get_json()
    assert "traceback" not in body
    assert "secret internal detail" not in body["error"]


def test_explain_falls_back_to_template_without_api_key(client):
    res = client.post(
        "/explain",
        json={"sentence": "The vendor shall indemnify.", "risk_type": "Liability", "severity": "High"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["source"] == "template"
    assert body["explanation"]


def test_explain_validates_input(client):
    assert client.post("/explain", json={"sentence": "x"}).status_code == 400
    long = {"sentence": "a" * 8001, "risk_type": "Liability", "severity": "High"}
    assert client.post("/explain", json=long).status_code == 400
