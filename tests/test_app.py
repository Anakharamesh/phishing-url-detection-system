"""Comprehensive automated test suite for Phishing URL Detection System.

Tests:
1. Feature extraction correctness (25 features, exact ordering, edge cases).
2. Model inference engine (Random Forest, Linear SVM, DNN, CNN).
3. Majority voting and consensus calculation.
4. FastAPI endpoints, error handling, and image serving.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from app.feature_extractor import FEATURE_NAMES, SUSPICIOUS_WORDS, extract_features
from app.model_engine import ModelEngine
from app.main import app
from starlette.testclient import TestClient


def test_feature_names_match():
    """Verify extracted feature names match feature_names.json exactly."""
    with open(PROJECT_ROOT / "feature_names.json", "r") as f:
        expected = json.load(f)
    assert FEATURE_NAMES == expected, "FEATURE_NAMES must match feature_names.json in order and count"
    assert len(FEATURE_NAMES) == 25, f"Expected 25 features, got {len(FEATURE_NAMES)}"
    print("PASS: test_feature_names_match")


def test_feature_extraction_normalization():
    """Verify URL normalization, scheme prepending, and edge cases."""
    # 1. Bare domain without scheme
    vec1, dict1 = extract_features("google.com/search")
    assert len(vec1) == 25
    assert dict1["is_https"] == 0.0
    assert dict1["hostname_length"] == len("google.com")
    assert dict1["path_length"] == len("/search")

    # 2. HTTPS scheme
    vec2, dict2 = extract_features("https://secure-bank.com")
    assert dict2["is_https"] == 1.0

    # 3. Subdomain counting: max(len(hostname.split(".")) - 2, 0)
    _, dict3 = extract_features("https://a.b.c.example.com")
    # hostname: a.b.c.example.com -> 5 parts -> 5 - 2 = 3
    assert dict3["subdomain_count"] == 3.0

    _, dict4 = extract_features("https://example.com")
    # hostname: example.com -> 2 parts -> 2 - 2 = 0
    assert dict4["subdomain_count"] == 0.0

    # 4. Suspicious words count
    _, dict5 = extract_features("http://login-update-account.bank.com/signin/verify")
    # "login", "update", "account", "bank", "signin", "verify" = 6 keywords
    assert dict5["has_suspicious_word"] == 1.0
    assert dict5["suspicious_word_count"] == 6.0

    # 5. Invalid / unusual port handling (should not crash)
    vec6, dict6 = extract_features("http://example.com:999999999/test")
    assert len(vec6) == 25
    assert dict6["has_port"] in (0.0, 1.0)

    print("PASS: test_feature_extraction_normalization")


def test_model_inference_and_voting():
    """Verify individual model predictions and majority voting rules."""
    engine = ModelEngine(base_dir=PROJECT_ROOT)

    # Legitimate URL test
    vec_legit, _ = extract_features("https://www.google.com")
    res_legit = engine.predict(vec_legit)
    assert res_legit["total_models"] == 4
    assert res_legit["overall_verdict"] == "LEGITIMATE"
    assert res_legit["phishing_votes"] < 2
    assert res_legit["is_phishing"] is False
    assert len(res_legit["models"]) == 4

    # Phishing URL test
    vec_phish, _ = extract_features("http://paypal-security-update-account.com/login/verify.php?user=admin")
    res_phish = engine.predict(vec_phish)
    assert res_phish["overall_verdict"] == "PHISHING"
    assert res_phish["phishing_votes"] >= 2
    assert res_phish["is_phishing"] is True
    assert res_phish["model_consensus_percentage"] >= 50.0

    # Verify each model returned expected keys
    for m in res_phish["models"]:
        assert "id" in m
        assert "name" in m
        assert "prediction" in m
        assert m["prediction"] in ("LEGITIMATE", "PHISHING")
        assert "phishing_score" in m
        assert "preprocessing" in m

    print("PASS: test_model_inference_and_voting")


def test_fastapi_endpoints():
    """Verify API client responses, status codes, and JSON contracts."""
    client = TestClient(app)

    # 1. GET /
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert "Phishing URL Detection System" in res_home.text

    # 2. GET /api/features
    res_feats = client.get("/api/features")
    assert res_feats.status_code == 200
    feats_data = res_feats.json()
    assert len(feats_data) == 25
    assert feats_data[0]["name"] == "url_length"

    # 3. GET /api/metrics
    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert len(metrics_data["models"]) == 4

    # 4. GET /images/...
    for img_name in ("model_accuracy_comparison.png", "confusion_matrix_comparison.png", "deep_learning_training_graphs.png"):
        res_img = client.get(f"/images/{img_name}")
        assert res_img.status_code == 200
        assert res_img.headers["content-type"].startswith("image/")

    # 5. POST /api/predict (Legitimate URL)
    res_pred_legit = client.post("/api/predict", json={"url": "https://www.wikipedia.org"})
    assert res_pred_legit.status_code == 200
    pred_data = res_pred_legit.json()
    assert pred_data["overall_verdict"] == "LEGITIMATE"
    assert len(pred_data["models"]) == 4
    assert len(pred_data["features"]) == 25

    # 6. POST /api/predict (Phishing URL)
    res_pred_phish = client.post("/api/predict", json={"url": "http://192.168.1.1/bank/signin/confirm/wallet/password.html"})
    assert res_pred_phish.status_code == 200
    pred_phish_data = res_pred_phish.json()
    assert pred_phish_data["overall_verdict"] == "PHISHING"

    # 7. Error handling: Empty URL
    res_empty = client.post("/api/predict", json={"url": "   "})
    assert res_empty.status_code in (400, 422)

    # 8. Error handling: Missing field
    res_missing = client.post("/api/predict", json={})
    assert res_missing.status_code == 422

    print("PASS: test_fastapi_endpoints")


if __name__ == "__main__":
    print("Running Test Suite...")
    test_feature_names_match()
    test_feature_extraction_normalization()
    test_model_inference_and_voting()
    test_fastapi_endpoints()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
