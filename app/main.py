"""FastAPI application for Phishing URL Detection.

Provides API endpoints for URL feature extraction, multi-model predictions,
and serves the responsive web interface.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.feature_extractor import FEATURE_NAMES, extract_features
from app.model_engine import ModelEngine
from app.schemas import FeatureMetadata, PredictionRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Phishing URL Detection System",
    description="Multi-model machine learning and deep learning phishing URL detection API.",
    version="1.0.0",
)

# Enable CORS for development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Engine
engine = ModelEngine(base_dir=BASE_DIR)

# Mount static files and templates
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FEATURE_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "url_length": ("Length / Dimension", "Total number of characters in the full URL string."),
    "hostname_length": ("Length / Dimension", "Length of the domain name / netloc component."),
    "path_length": ("Length / Dimension", "Length of the path following domain name."),
    "query_length": ("Length / Dimension", "Length of URL parameter query string."),
    "dot_count": ("Character Frequency", "Count of dot '.' characters in the URL."),
    "hyphen_count": ("Character Frequency", "Count of hyphen '-' characters in the URL."),
    "underscore_count": ("Character Frequency", "Count of underscore '_' characters in the URL."),
    "slash_count": ("Character Frequency", "Count of forward slash '/' characters in the URL."),
    "question_mark_count": ("Character Frequency", "Count of question mark '?' characters in the URL."),
    "equal_count": ("Character Frequency", "Count of equal sign '=' characters in the URL."),
    "at_count": ("Character Frequency", "Count of at '@' characters in the URL."),
    "ampersand_count": ("Character Frequency", "Count of ampersand '&' characters in the URL."),
    "percent_count": ("Character Frequency", "Count of percent '%' characters (URL encoding)."),
    "digit_count": ("Syntactic Distribution", "Total number of numeric digits [0-9] in URL."),
    "letter_count": ("Syntactic Distribution", "Total number of alphabetic letters [a-zA-Z] in URL."),
    "digit_ratio": ("Syntactic Distribution", "Ratio of digits to total URL length."),
    "letter_ratio": ("Syntactic Distribution", "Ratio of letters to total URL length."),
    "is_https": ("Protocol & Security", "Binary indicator (1 if HTTPS protocol, 0 otherwise)."),
    "is_ip_address": ("Domain Anomaly", "Binary indicator (1 if domain is raw IP address, 0 otherwise)."),
    "subdomain_count": ("Domain Hierarchy", "Number of subdomains parsed from the domain name."),
    "has_suspicious_word": ("Keyword Heuristic", "Binary indicator (1 if any phishing keyword is present, 0 otherwise)."),
    "suspicious_word_count": ("Keyword Heuristic", "Total frequency count of detected phishing keywords."),
    "has_port": ("Network Anomaly", "Binary indicator (1 if non-standard port is specified, 0 otherwise)."),
    "has_fragment": ("Structural Token", "Binary indicator (1 if fragment anchor '#' exists, 0 otherwise)."),
    "has_query": ("Structural Token", "Binary indicator (1 if query string '?' exists, 0 otherwise)."),
}


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> FileResponse:
    """Serve the single-page application frontend."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html template not found.",
        )
    return FileResponse(index_file)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_url(req: PredictionRequest) -> Dict[str, Any]:
    """Analyze a URL string and return predictions from all 4 models.

    Performs strictly lexical and structural URL analysis without visiting or fetching the URL.
    """
    raw_url = req.url.strip()
    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The submitted URL cannot be empty.",
        )

    try:
        feature_vector, feature_dict = extract_features(raw_url)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting URL features: {str(exc)}",
        )

    try:
        results = engine.predict(feature_vector)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during model prediction: {str(exc)}",
        )

    response = {
        "url": raw_url,
        "overall_verdict": results["overall_verdict"],
        "is_phishing": results["is_phishing"],
        "phishing_votes": results["phishing_votes"],
        "legitimate_votes": results["legitimate_votes"],
        "total_models": results["total_models"],
        "model_consensus_percentage": results["model_consensus_percentage"],
        "consensus_text": results["consensus_text"],
        "models": results["models"],
        "features": feature_dict,
    }
    return response


@app.get("/api/features", response_model=List[FeatureMetadata])
async def get_feature_metadata() -> List[Dict[str, Any]]:
    """Return documentation for each of the 25 extracted features."""
    items = []
    for idx, name in enumerate(FEATURE_NAMES):
        cat, desc = FEATURE_DESCRIPTIONS.get(name, ("General", "Extracted URL feature."))
        items.append({
            "index": idx + 1,
            "name": name,
            "category": cat,
            "description": desc,
        })
    return items


@app.get("/api/metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """Return model performance metrics and visualization links."""
    return {
        "models": [
            {"name": "Random Forest", "accuracy": 99.59, "type": "Ensemble (100 Trees)"},
            {"name": "Support Vector Machine", "accuracy": 99.38, "type": "LinearSVC"},
            {"name": "Deep Neural Network", "accuracy": 99.57, "type": "Dense Sequential (4 Layers)"},
            {"name": "1D Convolutional Neural Network", "accuracy": 99.57, "type": "Conv1D + Pooling Sequential"},
        ],
        "graphs": {
            "accuracy_comparison": "/images/model_accuracy_comparison.png",
            "confusion_matrices": "/images/confusion_matrix_comparison.png",
            "training_curves": "/images/deep_learning_training_graphs.png",
        },
    }


@app.get("/images/{image_name}")
async def serve_graph_image(image_name: str) -> FileResponse:
    """Serve existing evaluation plot images from workspace root."""
    valid_images = {
        "model_accuracy_comparison.png",
        "confusion_matrix_comparison.png",
        "deep_learning_training_graphs.png",
    }
    if image_name not in valid_images:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )
    img_path = BASE_DIR / image_name
    if not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested plot image file is missing.",
        )
    return FileResponse(img_path)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health status endpoint."""
    return {
        "status": "healthy",
        "loaded_models": ["RandomForest", "LinearSVC", "DNNModel", "CNNModel"],
        "scaler_features": 25,
    }
