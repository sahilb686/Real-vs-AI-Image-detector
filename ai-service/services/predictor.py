"""
Prediction orchestration service.

Receives a validated image buffer and model identifier, runs the appropriate
preprocessing pipeline and in-memory model, then returns a structured result.
"""

import logging
import time
from typing import BinaryIO

import numpy as np

from services.authenticity import AuthenticityResult, interpret_prediction
from services.model_loader import model_loader
from utils.preprocessing import (
    preprocess_cnn,
    preprocess_efficientnet,
    preprocess_efficientnet_art,
)

logger = logging.getLogger(__name__)


class PredictorService:
    """
    Coordinates preprocessing → inference → response formatting.

    All TensorFlow models are reused from the singleton ModelLoader;
    no models are built or loaded per request.
    """

    def __init__(self) -> None:
        self._models = model_loader.models

    def predict(self, image_buffer: BinaryIO, model_id: str) -> AuthenticityResult:
        """
        Run end-to-end prediction for the given image and model.

        Args:
            image_buffer: Seekable binary stream containing image bytes.
            model_id: Canonical model identifier (cnn, efficientnet, efficientnet-art).

        Returns:
            AuthenticityResult with prediction, probabilities, and timing metadata.
        """
        image_buffer.seek(0)
        start_time = time.perf_counter()

        # Step 1: Preprocess image into model input tensor (behaviour unchanged)
        img_array = self._preprocess(image_buffer, model_id)

        # Step 2: Run inference on the cached in-memory model
        raw_prediction = self._run_inference(img_array, model_id)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "Prediction complete model=%s raw_score=%.6f processing_time_ms=%d",
            model_id,
            float(raw_prediction[0][0]),
            processing_time_ms,
        )

        # Step 3: Interpret raw score using original threshold logic
        return interpret_prediction(raw_prediction, model_id, processing_time_ms)

    def _preprocess(self, image_buffer: BinaryIO, model_id: str) -> np.ndarray:
        """Dispatch to the correct preprocessing function for the selected model."""
        preprocessors = {
            "cnn": preprocess_cnn,
            "efficientnet": preprocess_efficientnet,
            "efficientnet-art": preprocess_efficientnet_art,
        }
        return preprocessors[model_id](image_buffer)

    def _run_inference(self, img_array: np.ndarray, model_id: str) -> np.ndarray:
        """
        Execute model.predict on the preprocessed array.

        EfficientNet models use the same TPU/default distribution strategy
        scope as the original Streamlit application.
        """
        if model_id == "cnn":
            return self._models.cnn_model.predict(img_array, verbose=0)

        strategy = self._models.distribution_strategy
        with strategy.scope():
            if model_id == "efficientnet":
                return self._models.efficientnet_model.predict(img_array, verbose=0)
            if model_id == "efficientnet-art":
                return self._models.efficientnet_art_model.predict(img_array, verbose=0)

        raise ValueError(f"Unsupported model_id: {model_id}")


def get_predictor() -> PredictorService:
    """Factory that returns a PredictorService bound to loaded models."""
    return PredictorService()
