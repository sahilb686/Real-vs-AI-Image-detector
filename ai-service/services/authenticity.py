"""
Authenticity interpretation service.

Converts raw sigmoid model output into structured prediction results.
Preserves the original classification threshold from the Streamlit app:
  - score < 0.5  → AI Generated
  - score >= 0.5 → REAL
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from services.metadata import get_model_display_name, get_model_version


@dataclass(frozen=True)
class AuthenticityResult:
    """Structured prediction result returned by the API."""

    prediction: str
    confidence: float
    real_probability: float
    ai_probability: float
    model: str
    model_version: str
    processing_time_ms: int
    timestamp: str
    success: bool

    def to_dict(self) -> dict:
        """Serialize the result as a JSON-compatible dictionary."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "real_probability": self.real_probability,
            "ai_probability": self.ai_probability,
            "model": self.model,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "success": self.success,
        }


def interpret_prediction(
    raw_prediction: np.ndarray,
    model_id: str,
    processing_time_ms: int,
) -> AuthenticityResult:
    """
    Map a raw model output array to an AuthenticityResult.

    The sigmoid output represents the probability of the image being REAL.
    Classification threshold matches the original Streamlit application.
    """
    # Extract scalar score — same indexing as original: predictions[0][0]
    score: float = float(raw_prediction[0][0])

    # Original threshold logic: predictions[0] < 0.5 → "AI Generated", else "REAL"
    is_real = score >= 0.5
    prediction_label = "REAL" if is_real else "AI"

    real_probability = round(score * 100, 1)
    ai_probability = round((1.0 - score) * 100, 1)

    # Confidence is the probability assigned to the predicted class
    confidence = real_probability if is_real else ai_probability

    return AuthenticityResult(
        prediction=prediction_label,
        confidence=confidence,
        real_probability=real_probability,
        ai_probability=ai_probability,
        model=get_model_display_name(model_id),
        model_version=get_model_version(model_id),
        processing_time_ms=processing_time_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
        success=True,
    )
