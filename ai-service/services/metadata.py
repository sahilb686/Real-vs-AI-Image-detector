"""
Model metadata service.

Provides display names, versions, and lookup helpers for supported models.
Keeps response formatting decoupled from raw model identifiers.
"""

from fastapi import HTTPException

from app.config import MODEL_METADATA, SUPPORTED_MODELS


def validate_model_identifier(model: str) -> str:
    """
    Normalize and validate a model identifier from the API request.

    Returns the canonical lowercase identifier.
    Raises HTTP 404 when the model is not supported.
    """
    normalized = model.strip().lower()

    if normalized not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model}' not found. "
                f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}."
            ),
        )

    return normalized


def get_model_display_name(model_id: str) -> str:
    """Return the human-readable model name for API responses."""
    return MODEL_METADATA[model_id]["display_name"]


def get_model_version(model_id: str) -> str:
    """Return the version string for a given model."""
    return MODEL_METADATA[model_id]["version"]
