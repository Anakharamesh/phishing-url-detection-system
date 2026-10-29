"""Model inference engine for Phishing URL Detection.

Integrates Random Forest, Linear SVM, Deep Neural Network (DNN),
and 1D Convolutional Neural Network (CNN) from pre-trained model files.
"""

from __future__ import annotations

import io
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd

from app.feature_extractor import FEATURE_NAMES

# Suppress sklearn unpickle version warnings for clean runtime logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class DNNModel:
    """Inference runner for the trained Keras Sequential DNN model.
    Architecture:
      Input (25) -> Dense(128, relu) -> Dropout
      -> Dense(64, relu) -> Dropout -> Dense(32, relu) -> Dense(1, sigmoid)
    """

    def __init__(self, keras_path: str | Path):
        self.path = str(keras_path)
        with zipfile.ZipFile(self.path, "r") as z:
            weights_bytes = z.read("model.weights.h5")
        with h5py.File(io.BytesIO(weights_bytes), "r") as f:
            self.w0 = np.array(f["layers/dense/vars/0"], dtype=np.float32)      # (25, 128)
            self.b0 = np.array(f["layers/dense/vars/1"], dtype=np.float32)      # (128,)
            self.w1 = np.array(f["layers/dense_1/vars/0"], dtype=np.float32)    # (128, 64)
            self.b1 = np.array(f["layers/dense_1/vars/1"], dtype=np.float32)    # (64,)
            self.w2 = np.array(f["layers/dense_2/vars/0"], dtype=np.float32)    # (64, 32)
            self.b2 = np.array(f["layers/dense_2/vars/1"], dtype=np.float32)    # (32,)
            self.w3 = np.array(f["layers/dense_3/vars/0"], dtype=np.float32)    # (32, 1)
            self.b3 = np.array(f["layers/dense_3/vars/1"], dtype=np.float32)    # (1,)

    def predict_proba(self, x: np.ndarray) -> float:
        """Forward pass for (1, 25) scaled feature input. Returns float probability."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h1 = _relu(np.dot(x, self.w0) + self.b0)
        h2 = _relu(np.dot(h1, self.w1) + self.b1)
        h3 = _relu(np.dot(h2, self.w2) + self.b2)
        out = _sigmoid(np.dot(h3, self.w3) + self.b3)
        return float(out[0, 0])


class CNNModel:
    """Inference runner for the trained Keras Sequential 1D CNN model.
    Architecture:
      Input (25, 1) -> Conv1D(64, k=3, same, relu) -> MaxPool1D(2, valid)
      -> Conv1D(128, k=3, same, relu) -> MaxPool1D(2, valid)
      -> Flatten(768) -> Dense(64, relu) -> Dropout -> Dense(1, sigmoid)
    """

    def __init__(self, keras_path: str | Path):
        self.path = str(keras_path)
        with zipfile.ZipFile(self.path, "r") as z:
            weights_bytes = z.read("model.weights.h5")
        with h5py.File(io.BytesIO(weights_bytes), "r") as f:
            self.w_conv1 = np.array(f["layers/conv1d/vars/0"], dtype=np.float32)     # (3, 1, 64)
            self.b_conv1 = np.array(f["layers/conv1d/vars/1"], dtype=np.float32)     # (64,)
            self.w_conv2 = np.array(f["layers/conv1d_1/vars/0"], dtype=np.float32)   # (3, 64, 128)
            self.b_conv2 = np.array(f["layers/conv1d_1/vars/1"], dtype=np.float32)   # (128,)
            self.w_dense1 = np.array(f["layers/dense/vars/0"], dtype=np.float32)     # (768, 64)
            self.b_dense1 = np.array(f["layers/dense/vars/1"], dtype=np.float32)     # (64,)
            self.w_dense2 = np.array(f["layers/dense_1/vars/0"], dtype=np.float32)   # (64, 1)
            self.b_dense2 = np.array(f["layers/dense_1/vars/1"], dtype=np.float32)   # (1,)

    @staticmethod
    def _conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        # x: (N, L, C_in), kernel: (K, C_in, C_out), bias: (C_out,)
        N, L, C_in = x.shape
        K, _, C_out = kernel.shape
        pad = K // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant", constant_values=0.0)
        out = np.zeros((N, L, C_out), dtype=np.float32)
        for i in range(L):
            window = x_padded[:, i : i + K, :]
            out[:, i, :] = np.tensordot(window, kernel, axes=([1, 2], [0, 1])) + bias
        return out

    @staticmethod
    def _maxpool1d(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        N, L, C = x.shape
        out_len = (L - pool_size) // stride + 1
        out = np.zeros((N, out_len, C), dtype=np.float32)
        for i in range(out_len):
            start = i * stride
            out[:, i, :] = np.max(x[:, start : start + pool_size, :], axis=1)
        return out

    def predict_proba(self, x: np.ndarray) -> float:
        """Forward pass for (1, 25, 1) scaled feature input. Returns float probability."""
        if x.ndim == 1:
            x = x.reshape(1, 25, 1)
        elif x.ndim == 2:
            x = x[:, :, np.newaxis]

        c1 = _relu(self._conv1d_same(x, self.w_conv1, self.b_conv1))
        p1 = self._maxpool1d(c1, pool_size=2, stride=2)

        c2 = _relu(self._conv1d_same(p1, self.w_conv2, self.b_conv2))
        p2 = self._maxpool1d(c2, pool_size=2, stride=2)

        flat = p2.reshape((p2.shape[0], -1))
        d1 = _relu(np.dot(flat, self.w_dense1) + self.b_dense1)
        out = _sigmoid(np.dot(d1, self.w_dense2) + self.b_dense2)
        return float(out[0, 0])


class ModelEngine:
    """Central engine managing model loading, transformations, and predictions."""

    def __init__(self, base_dir: str | Path = "."):
        self.base_dir = Path(base_dir)
        self.scaler = joblib.load(self.base_dir / "scaler.pkl")
        self.rf = joblib.load(self.base_dir / "random_forest.pkl")
        self.svm = joblib.load(self.base_dir / "svm.pkl")
        self.dnn = DNNModel(self.base_dir / "dnn_model.keras")
        self.cnn = CNNModel(self.base_dir / "cnn_model.keras")

    def predict(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Run inference on all 4 models and return combined predictions."""
        # Wrap into DataFrame with exact feature names to match training schema
        raw_df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
        raw_x = raw_df.to_numpy(dtype=np.float32)

        # Scaled features array: shape (1, 25)
        scaled_x = self.scaler.transform(raw_df)
        scaled_df = pd.DataFrame(scaled_x, columns=FEATURE_NAMES)

        # 1. Random Forest (receives raw features)
        rf_pred_val = int(self.rf.predict(raw_df)[0])
        rf_probs = getattr(self.rf, "predict_proba", lambda _: [[0.5, 0.5]])(raw_df)[0]
        rf_phish_prob = float(rf_probs[1]) if len(rf_probs) > 1 else float(rf_pred_val)

        # 2. SVM (receives scaled features)
        svm_pred_val = int(self.svm.predict(scaled_x)[0])
        svm_decision = float(self.svm.decision_function(scaled_x)[0])
        # Approximate sigmoid score from distance for UI visualization
        svm_score = float(_sigmoid(np.array([svm_decision]))[0])

        # 3. DNN (receives scaled features)
        dnn_phish_prob = self.dnn.predict_proba(scaled_x)
        dnn_pred_val = 1 if dnn_phish_prob >= 0.5 else 0

        # 4. CNN (receives scaled features reshaped to (1, 25, 1))
        cnn_x = scaled_x.reshape(1, 25, 1)
        cnn_phish_prob = self.cnn.predict_proba(cnn_x)
        cnn_pred_val = 1 if cnn_phish_prob >= 0.5 else 0

        models_data = [
            {
                "id": "random_forest",
                "name": "Random Forest",
                "type": "Machine Learning Ensemble",
                "preprocessing": "Raw 25 Features",
                "accuracy": "99.59%",
                "prediction": "PHISHING" if rf_pred_val == 1 else "LEGITIMATE",
                "is_phishing": bool(rf_pred_val == 1),
                "phishing_score": round(rf_phish_prob * 100, 2),
                "metric_label": "Probability",
            },
            {
                "id": "svm",
                "name": "Support Vector Machine (Linear)",
                "type": "Linear Classifier",
                "preprocessing": "StandardScaled (25)",
                "accuracy": "99.38%",
                "prediction": "PHISHING" if svm_pred_val == 1 else "LEGITIMATE",
                "is_phishing": bool(svm_pred_val == 1),
                "phishing_score": round(svm_score * 100, 2),
                "decision_margin": round(svm_decision, 4),
                "metric_label": "Margin Confidence",
            },
            {
                "id": "dnn",
                "name": "Deep Neural Network (DNN)",
                "type": "Multi-Layer Perceptron (4 Dense Layers)",
                "preprocessing": "StandardScaled (25)",
                "accuracy": "99.57%",
                "prediction": "PHISHING" if dnn_pred_val == 1 else "LEGITIMATE",
                "is_phishing": bool(dnn_pred_val == 1),
                "phishing_score": round(dnn_phish_prob * 100, 2),
                "metric_label": "Probability",
            },
            {
                "id": "cnn",
                "name": "1D Convolutional Neural Network (CNN)",
                "type": "Sequential 1D CNN (2 Conv1D + MaxPool)",
                "preprocessing": "StandardScaled & Reshaped (25, 1)",
                "accuracy": "99.57%",
                "prediction": "PHISHING" if cnn_pred_val == 1 else "LEGITIMATE",
                "is_phishing": bool(cnn_pred_val == 1),
                "phishing_score": round(cnn_phish_prob * 100, 2),
                "metric_label": "Probability",
            },
        ]

        # Majority voting: at least 2 models voting PHISHING -> PHISHING
        votes = [rf_pred_val, svm_pred_val, dnn_pred_val, cnn_pred_val]
        phishing_votes = sum(votes)
        legitimate_votes = 4 - phishing_votes
        consensus_percentage = round((phishing_votes / 4.0) * 100.0, 1)

        is_overall_phishing = phishing_votes >= 2
        overall_verdict = "PHISHING" if is_overall_phishing else "LEGITIMATE"

        return {
            "overall_verdict": overall_verdict,
            "is_phishing": is_overall_phishing,
            "phishing_votes": phishing_votes,
            "legitimate_votes": legitimate_votes,
            "total_models": 4,
            "model_consensus_percentage": consensus_percentage,
            "consensus_text": f"{phishing_votes} of 4 models classify this URL as PHISHING ({consensus_percentage}% consensus)",
            "models": models_data,
        }
