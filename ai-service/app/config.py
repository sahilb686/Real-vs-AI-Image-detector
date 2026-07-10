"""
Application configuration module.

Centralizes all environment-driven settings, model paths, and API constants
so the rest of the service remains decoupled from filesystem layout.
"""

import os
from pathlib import Path
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

# Project root is one level above the `app/` package directory.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Model file paths (weights are never modified — only loaded at startup)
# ---------------------------------------------------------------------------

CNN_WEIGHTS_PATH: Path = MODELS_DIR / "model_weights.weights.h5"
EFFICIENTNET_MODEL_PATH: Path = MODELS_DIR / "efficientnetb3_binary_classifier_8.h5"
EFFICIENTNET_ART_MODEL_PATH: Path = MODELS_DIR / "EfficientNet_fine_tune_art_model.h5"

# ---------------------------------------------------------------------------
# API / server settings
# ---------------------------------------------------------------------------

APP_TITLE: str = "AI Image Authenticity Service"
APP_DESCRIPTION: str = (
    "Production REST API for detecting whether an image is real or AI-generated. "
    "Supports CNN, EfficientNetB3, and EfficientNet Art classifiers."
)
APP_VERSION: str = "1.0.0"

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Supported models and file types
# ---------------------------------------------------------------------------

# API model identifiers accepted in the `model` form field.
SUPPORTED_MODELS: FrozenSet[str] = frozenset({"cnn", "efficientnet", "efficientnet-art"})

# Image extensions allowed for upload validation.
ALLOWED_IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png"})
ALLOWED_IMAGE_MIME_TYPES: FrozenSet[str] = frozenset(
    {"image/jpeg", "image/jpg", "image/png"}
)

# Maximum upload size in bytes (default 10 MB).
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))

# ---------------------------------------------------------------------------
# Model metadata (display names and versions returned in API responses)
# ---------------------------------------------------------------------------

MODEL_METADATA: dict[str, dict[str, str]] = {
    "cnn": {
        "display_name": "CNN",
        "version": "1.0.0",
    },
    "efficientnet": {
        "display_name": "EfficientNetB3",
        "version": "1.0.0",
    },
    "efficientnet-art": {
        "display_name": "EfficientNet Art",
        "version": "1.0.0",
    },
}
