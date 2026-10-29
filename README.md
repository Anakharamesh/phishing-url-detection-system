# Phishing URL Detection Using Machine Learning and Deep Learning

A comprehensive, production-grade cybersecurity web application that detects phishing URLs using four trained Machine Learning and Deep Learning models: **Random Forest**, **Linear Support Vector Machine (SVM)**, **Deep Neural Network (DNN)**, and a **1D Convolutional Neural Network (CNN)**.

The system performs strictly **lexical and structural analysis** directly on the submitted URL string—making zero external network or HTTP requests to guarantee fast, safe, and air-gapped threat detection.

---
## 👩‍💻 Team Members

| Name | Student ID |
|------|------------|
| Anakha Ramesh | KU2407U798 |
| Sanchita Girgilani | KU2407U777 |

**Project:** Phishing URL Detection System Using Machine Learning and Deep Learning

**Academic Project:** Deep Learning Mini Project

## Key Features

* **Multi-Model Consensus Engine**: Integrates 4 distinct classifiers trained on phishing URL datasets:
  * **Random Forest** (99.59% Test Accuracy) — Evaluated on 25 raw lexical features.
  * **Linear SVM** (99.38% Test Accuracy) — Evaluated on 25 StandardScaler-normalized features.
  * **Deep Neural Network (DNN)** (99.57% Test Accuracy) — 4-layer MLP with ReLU and Dropout, evaluated on scaled features.
  * **1D Convolutional Neural Network (CNN)** (99.57% Test Accuracy) — Dual Conv1D + MaxPooling architecture evaluated on reshaped `(batch, 25, 1)` scaled tensors.
* **Strict 25 Lexical & Structural Features**: Faithfully replicates the exact feature extraction logic and ordering from training (`feature_names.json`).
* **Majority Voting & Consensus**: If at least 2 out of 4 models vote **PHISHING**, the overall security classification is **PHISHING**; otherwise **LEGITIMATE**. The consensus percentage reflects inter-model agreement.
* **Modern Responsive Web UI**: High-contrast cyber-themed single page application built with responsive design, quick test chips, animated loading indicators, individual model cards, an interactive 25-feature inspection table, and embedded training/evaluation graphs.
* **High Performance FastAPI Backend**: Asynchronous REST API with Pydantic request validation, thorough error handling, and automated test coverage.

---

## 25 URL Features Extracted

Features are extracted in the exact order specified by `feature_names.json`:

| # | Feature Name | Description | Extraction / Normalization Rule |
|---|:---|:---|:---|
| 1 | `url_length` | Total URL length | `len(url)` |
| 2 | `hostname_length` | Domain / host length | `len(parsed.hostname)` |
| 3 | `path_length` | Path length | `len(parsed.path)` |
| 4 | `query_length` | Query string length | `len(parsed.query)` |
| 5 | `dot_count` | Dot `.` count | Frequency in full URL string |
| 6 | `hyphen_count` | Hyphen `-` count | Frequency in full URL string |
| 7 | `underscore_count`| Underscore `_` count | Frequency in full URL string |
| 8 | `slash_count` | Slash `/` count | Frequency in full URL string |
| 9 | `question_mark_count` | `?` count | Frequency in full URL string |
| 10 | `equal_count` | `=` count | Frequency in full URL string |
| 11 | `at_count` | `@` symbol count | Frequency in full URL string |
| 12 | `ampersand_count` | `&` symbol count | Frequency in full URL string |
| 13 | `percent_count` | `%` count (encoded chars) | Frequency in full URL string |
| 14 | `digit_count` | Numeric digit count | Count of `[0-9]` in URL |
| 15 | `letter_count` | Alphabetic character count | Count of `[a-zA-Z]` in URL |
| 16 | `digit_ratio` | Ratio of digits to length | `digit_count / url_length` |
| 17 | `letter_ratio` | Ratio of letters to length | `letter_count / url_length` |
| 18 | `is_https` | HTTPS scheme indicator | 1 if starts with `https://`, 0 otherwise |
| 19 | `is_ip_address` | Host is raw IP address | 1 if host matches IPv4 format, 0 otherwise |
| 20 | `subdomain_count` | Number of subdomains | `max(len(hostname.split(".")) - 2, 0)` |
| 21 | `has_suspicious_word` | Phishing keyword indicator | 1 if any suspicious word present, 0 otherwise |
| 22 | `suspicious_word_count` | Phishing keyword frequency | Occurrences of words: `["login", "verify", "secure", "account", "update", "bank", "signin", "password", "confirm", "payment", "wallet"]` |
| 23 | `has_port` | Explicit port specified | `int(parsed.port is not None)` (safely handled) |
| 24 | `has_fragment` | URL anchor fragment present | 1 if `#fragment` exists, 0 otherwise |
| 25 | `has_query` | Query parameters present | 1 if `?query` exists, 0 otherwise |

---

## Project Structure

```
phishing_detection_project/
├── app/
│   ├── __init__.py
│   ├── feature_extractor.py       # 25 URL lexical features extraction
│   ├── model_engine.py            # Model loading, transforms & forward passes
│   ├── schemas.py                 # Pydantic validation schemas
│   └── main.py                    # FastAPI server & route handlers
├── static/
│   ├── css/
│   │   └── style.css              # Cyber-themed responsive styling
│   └── js/
│       └── app.js                 # Frontend API client & interactive UI
├── templates/
│   └── index.html                 # Single page application template
├── tests/
│   └── test_app.py                # Automated end-to-end test suite
├── random_forest.pkl              # Saved trained Random Forest model
├── svm.pkl                        # Saved trained Linear SVM model
├── dnn_model.keras                # Saved trained Deep Neural Network
├── cnn_model.keras                # Saved trained 1D Convolutional Neural Network
├── scaler.pkl                     # Saved trained StandardScaler
├── feature_names.json             # 25 feature schema
├── model_accuracy_comparison.png  # Accuracy evaluation chart
├── confusion_matrix_comparison.png# Confusion matrices chart
├── deep_learning_training_graphs.png # Training loss & accuracy curves
├── requirements.txt               # Python package dependencies
└── README.md                      # Documentation
```

---

## Installation & Setup Instructions

### 1. Prerequisites
* Python 3.10+ (or Anaconda / Miniconda)

### 2. Install Dependencies
Using pip or your preferred Python environment:
```bash
pip install -r requirements.txt
```

*(If using an existing Anaconda environment, activate it and install `fastapi` and `uvicorn`:)*
```bash
conda activate base
pip install fastapi uvicorn
```

---

## Running the Application

Start the web application server with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once started, open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Running Automated Tests

Run the comprehensive test suite verifying feature extraction, all 4 models, consensus voting, and API endpoints:

```bash
python tests/test_app.py
```

All 4 test phases should pass:
* `PASS: test_feature_names_match`
* `PASS: test_feature_extraction_normalization`
* `PASS: test_model_inference_and_voting`
* `PASS: test_fastapi_endpoints`

---

## API Endpoints

### 1. `POST /api/predict`
Analyzes a URL string and returns individual and ensemble classifications.

**Request Body:**
```json
{
  "url": "http://paypal-security-update-account.com/login/verify.php?user=admin"
}
```

**Response (Sample):**
```json
{
  "url": "http://paypal-security-update-account.com/login/verify.php?user=admin",
  "overall_verdict": "PHISHING",
  "is_phishing": true,
  "phishing_votes": 4,
  "legitimate_votes": 0,
  "total_models": 4,
  "model_consensus_percentage": 100.0,
  "consensus_text": "4 of 4 models classify this URL as PHISHING (100.0% consensus)",
  "models": [
    {
      "id": "random_forest",
      "name": "Random Forest",
      "type": "Machine Learning Ensemble",
      "preprocessing": "Raw 25 Features",
      "accuracy": "99.59%",
      "prediction": "PHISHING",
      "is_phishing": true,
      "phishing_score": 100.0,
      "metric_label": "Probability"
    },
    {
      "id": "svm",
      "name": "Support Vector Machine (Linear)",
      "type": "Linear Classifier",
      "preprocessing": "StandardScaled (25)",
      "accuracy": "99.38%",
      "prediction": "PHISHING",
      "is_phishing": true,
      "phishing_score": 100.0,
      "decision_margin": 13.82,
      "metric_label": "Margin Confidence"
    },
    {
      "id": "dnn",
      "name": "Deep Neural Network (DNN)",
      "type": "Multi-Layer Perceptron (4 Dense Layers)",
      "preprocessing": "StandardScaled (25)",
      "accuracy": "99.57%",
      "prediction": "PHISHING",
      "is_phishing": true,
      "phishing_score": 100.0,
      "metric_label": "Probability"
    },
    {
      "id": "cnn",
      "name": "1D Convolutional Neural Network (CNN)",
      "type": "Sequential 1D CNN (2 Conv1D + MaxPool)",
      "preprocessing": "StandardScaled & Reshaped (25, 1)",
      "accuracy": "99.57%",
      "prediction": "PHISHING",
      "is_phishing": true,
      "phishing_score": 100.0,
      "metric_label": "Probability"
    }
  ],
  "features": {
    "url_length": 71.0,
    "hostname_length": 34.0,
    "path_length": 17.0,
    "query_length": 10.0,
    "dot_count": 2.0,
    "hyphen_count": 3.0,
    ...
  }
}
```

### 2. `GET /api/features`
Returns complete documentation and categories for the 25 features.

### 3. `GET /api/metrics`
Returns accuracy comparison data and links to the visual evaluation plots.

### 4. `GET /images/{image_name}`
Serves the project's evaluation graphics (`model_accuracy_comparison.png`, `confusion_matrix_comparison.png`, `deep_learning_training_graphs.png`).

